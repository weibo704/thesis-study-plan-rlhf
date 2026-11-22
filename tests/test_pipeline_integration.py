"""End-to-end pipeline tests against synthetic fixtures.

Runs stages 1-5 in a scratch data root via the real CLI, then asserts the
invariants the plan depends on. Three of these are regression tests for bugs
found during the first fixture run.
"""
from __future__ import annotations

import csv
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
sys.path.insert(0, str(REPO))
from tests import make_fixtures  # noqa: E402


def _run_pipeline(root: Path) -> None:
    env = dict(os.environ, THESIS_DATA_ROOT=str(root), PYTHONPATH=str(REPO))
    proc = subprocess.run(
        [str(PYTHON), "-m", "pipeline.cli", "run", "--to", "5"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"pipeline failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("build")
    make_fixtures.build(root)
    _run_pipeline(root)
    return root


@pytest.fixture(scope="module")
def deduped(built: Path) -> pd.DataFrame:
    return pd.read_parquet(built / "data" / "staging" / "deduped.parquet")


class TestIdentityInvariants:
    def test_course_id_is_unique(self, deduped):
        """Regression: two source rows sharing a url_slug produced a
        duplicate course_id, which broke canonical-row selection."""
        assert deduped["course_id"].is_unique

    def test_exactly_one_canonical_row_per_cluster(self, deduped):
        per_cluster = deduped.groupby("dedup_cluster_id")["is_canonical"].sum()
        assert set(per_cluster.unique()) == {1}, dict(per_cluster[per_cluster != 1])

    def test_duplicate_rows_collapse_into_one_cluster(self, deduped):
        pytorch = deduped[deduped["title_key"].str.contains("pytorch", na=False)]
        assert len(pytorch) == 2
        assert pytorch["dedup_cluster_id"].nunique() == 1


class TestDonorDiscipline:
    def test_donors_never_create_a_course_row(self, deduped):
        sources = set(deduped["source_id"])
        assert "coursera2025" not in sources
        assert "coursera_hours" not in sources

    def test_slug_donor_supplies_real_effort_hours(self, deduped):
        """coursera_hours joins on url_slug exactly and is the only source of
        effort hours for Coursera, which otherwise has none."""
        cs = deduped[deduped["source_id"] == "coursera2021"]
        assert cs["effort_hours_total"].notna().any()

    def test_donor_row_with_no_spine_slug_is_ignored(self, deduped):
        assert not deduped["title_key"].str.contains("fashion", na=False).any()

    def test_donor_fills_a_gap_and_is_recorded(self, built, deduped):
        prov = pd.read_parquet(built / "data" / "catalog" / "field_provenance.parquet")
        donated = prov[prov["source_id"] == "coursera2025"]
        assert len(donated) > 0
        assert set(donated["transform"]) == {"donor_fill"}

    def test_declared_donor_level_beats_stage3_inference(self, deduped):
        """Regression: stage 3's text inference permanently shadowed a real
        declared level arriving from the donor."""
        row = deduped[deduped["title_key"].str.contains("sql", na=False)].iloc[0]
        assert row["level_source"] == "declared"
        assert bool(row["level_is_imputed"]) is False


class TestNullAndSentinelHandling:
    def test_no_null_lookalike_strings_leak_into_text(self, deduped):
        """Regression: float('nan') was str()'d into the literal 'nan'."""
        for col in ("description", "syllabus", "title_normalized", "institution"):
            values = deduped[col].dropna().astype(str).str.strip().str.lower()
            assert not values.isin({"nan", "none", "<na>", ""}).any(), col

    def test_missing_description_stays_null(self, deduped):
        assert deduped["description"].isna().any()

    def test_missing_rating_is_flagged_not_zeroed(self, deduped):
        missing = deduped[deduped["rating_is_missing"].astype(bool)]
        assert len(missing) > 0
        assert missing["rating_mean"].isna().all()

    def test_every_imputed_level_carries_its_flag(self, deduped):
        inferred = deduped[deduped["level_source"] == "inferred_text"]
        assert inferred["level_is_imputed"].astype(bool).all()
        declared = deduped[deduped["level_source"] == "declared"]
        assert not declared["level_is_imputed"].astype(bool).any()


class TestDurationSeparation:
    def test_month_buckets_never_become_effort_hours(self, deduped):
        months = deduped[deduped["duration_unit_original"].fillna("").str.contains("Month")]
        assert len(months) > 0
        assert months["duration_lower_bound"].isna().all()
        assert months["effort_hours_total"].isna().all()
        assert months["calendar_weeks_lo"].notna().all()

    def test_effort_total_only_where_genuinely_derivable(self, deduped):
        have_total = deduped[deduped["effort_hours_total"].notna()]
        assert set(have_total["duration_source"]) <= {"derived_effort_x_weeks", "declared"}

    def test_missing_duration_stays_null(self, deduped):
        """A spine row matched by no donor must keep a null duration rather
        than having one invented."""
        rows = deduped[deduped["title_key"].str.contains("transformers", na=False)]
        assert len(rows), "fixture row missing"
        row = rows.iloc[0]
        assert pd.isna(row["effort_hours_total"])
        assert row["duration_source"] == "unknown"

    def test_intervals_are_preserved(self, deduped):
        buckets = deduped[deduped["duration_source"] == "bucket"]
        widths = (buckets["calendar_weeks_hi"] - buckets["calendar_weeks_lo"]).dropna()
        assert (widths > 0).any(), "bucket intervals collapsed to points"


class TestEncodingAndContainers:
    def test_mojibake_is_repaired(self, deduped):
        institutions = set(deduped["institution"].dropna())
        assert "École Polytechnique" in institutions
        assert not any("Ã" in i for i in institutions)

    def test_containers_are_labelled_as_programs(self, deduped):
        spec = deduped[deduped["title_key"].str.contains("specialization", na=False)]
        assert (spec["content_granularity"] == "program").all()

    def test_language_is_labelled_not_dropped(self, deduped):
        """Scope filtering is stage 8's job; stage 3 must only label."""
        assert deduped["language"].notna().all()

    def test_only_enabled_sources_reach_the_spine(self, deduped):
        """edX is disabled; its fixture exists but must not create rows."""
        assert "edx" not in set(deduped["source_id"])


class TestReproducibility:
    def test_rebuild_is_byte_identical(self, tmp_path_factory):
        hashes = []
        for _ in range(2):
            root = tmp_path_factory.mktemp("repeat")
            make_fixtures.build(root)
            _run_pipeline(root)
            df = pd.read_parquet(root / "data" / "staging" / "deduped.parquet")
            sys.path.insert(0, str(REPO))
            from pipeline.hashing import sha256_frame
            hashes.append(sha256_frame(df))
        assert hashes[0] == hashes[1], "pipeline is not deterministic"

    def test_dedup_is_input_order_independent(self, tmp_path_factory):
        """Shuffling source rows must not change the clustering.

        This is the same invariant the plan demands of the taxonomy build:
        order-dependence is what makes an incremental design irreproducible.
        """
        groupings = []
        for seed in (0, 7):
            root = tmp_path_factory.mktemp(f"shuffled{seed}")
            make_fixtures.build(root)
            path = root / "data" / "raw" / "coursera2021" / "Coursera.csv"
            with open(path, encoding="cp1252", newline="") as fh:
                rows = list(csv.reader(fh))
            header, body = rows[0], rows[1:]
            random.Random(seed).shuffle(body)
            with open(path, "w", encoding="cp1252", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerows(body)
            _run_pipeline(root)
            df = pd.read_parquet(root / "data" / "staging" / "deduped.parquet")
            groupings.append(sorted(
                tuple(sorted(g)) for g in
                df.groupby("dedup_cluster_id")["title_key"].apply(list)
            ))
        assert groupings[0] == groupings[1]


class TestProvenanceAndManifest:
    def test_sources_table_records_licence_and_hash(self, built):
        src = pd.read_parquet(built / "data" / "catalog" / "sources.parquet")
        # edX is disabled (see config/sources.yaml), so it never reaches
        # the sources table even though its fixture exists.
        assert set(src["source_id"]) == {"coursera2021", "coursera2025",
                                          "coursera_hours"}
        assert src["file_sha256"].str.len().eq(64).all()
        assert src["license"].notna().all()
        # The non-redistributable flag is what gates raw-text export later.
        assert not bool(src.set_index("source_id").loc["coursera2021", "redistributable"])

    def test_manifest_records_every_artifact(self, built):
        import json
        manifest = json.loads((built / "reports" / "manifest.json").read_text())
        arts = manifest["artifacts"]
        assert any("deduped.parquet" in k for k in arts)
        assert all(len(v["sha256"]) == 64 for v in arts.values())
        assert {"fetch", "parse", "normalize", "merge", "dedup"} <= set(manifest["stages"])


class TestEncodingDetection:
    """Regression: detect_encoding required the decoded text to contain no
    U+FFFD. The real Coursera export is valid UTF-8 that already contains
    baked-in U+FFFD, so that rule rejected UTF-8 and fell through to cp1252,
    which cannot raise and therefore mis-decoded the entire file."""

    def test_prefers_first_codec_that_decodes(self, tmp_path):
        from pipeline.stage02_parse import detect_encoding
        f = tmp_path / "damaged.csv"
        # Valid UTF-8 containing a genuine replacement char plus an accent.
        f.write_bytes("title\nContent�s Reach — École\n".encode("utf-8"))
        encoding, ok = detect_encoding(f)
        assert encoding == "utf-8" and ok is True

    def test_damage_is_reported_not_silently_recoded(self, built):
        import pandas as pd
        report = pd.read_csv(built / "reports" / "encoding_report.csv")
        assert set(report["encoding"]) <= {"utf-8", "utf-8-sig", "cp1252", "latin-1"}
        # The counter must measure post-repair damage, which is what lands
        # in the catalog; counting pre-repair understates it.
        assert "cells_with_replacement_char" in report.columns
