"""Text repair, normalisation keys, and union-find.

Three jobs, all deterministic and dependency-light so that the merge and
dedup stages are reproducible by construction:

  * repair_text  -- undo mojibake and normalise unicode
  * merge_key    -- order-insensitive blocking key for near-duplicate names
  * UnionFind    -- transitive closure over pairwise merge decisions

The mojibake repair is not cosmetic. The Coursera 2021 export mangles
non-ASCII institution names ("Ecole Polytechnique" arrives corrupted), and
because the corruption is concentrated in non-English institutions, leaving
it in place biases dedup and embeddings non-uniformly.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlsplit

import ftfy

# Dropped from merge keys: they carry no disambiguating signal in course or
# skill names and their presence is inconsistent across sources.
_STOPWORDS = frozenset(
    """
    a an the of for in to and or with on at by from as into via using use
    your you introduction intro course courses specialization guided project
    """.split()
)

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NONSLUG_RE = re.compile(r"[^a-z0-9]+")


def repair_text(value: Optional[str]) -> Optional[str]:
    """Fix mojibake, normalise to NFC, collapse whitespace.

    Returns None for null-ish input so that downstream null handling stays
    explicit rather than silently producing empty strings.
    """
    if value is None:
        return None
    # float('nan') is not None and str()s to "nan", which would silently
    # become a literal description. Catch every null flavour here.
    if isinstance(value, float) and value != value:
        return None
    text = str(value)
    if not text.strip() or text.strip().lower() in ("nan", "<na>", "none"):
        return None
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text or None


def normalize_title(value: Optional[str]) -> Optional[str]:
    """Human-readable canonical title: repaired, NFC, whitespace-collapsed.

    Case is preserved -- this is the display form. Use merge_key for matching.
    """
    return repair_text(value)


def merge_key(value: Optional[str]) -> str:
    """Order-insensitive matching key.

    casefold -> NFKD -> strip accents -> strip punctuation -> drop stopwords
    -> sort tokens -> hyphen-join.

    Sorting the tokens is the highest-yield step: it collapses "automated
    testing" and "testing automated" onto one key, which is the dominant
    duplicate pattern in generated skill labels.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT_RE.sub(" ", text)
    tokens = [t for t in _WS_RE.sub(" ", text).split() if t and t not in _STOPWORDS]
    if not tokens:
        # Everything was a stopword; fall back to the unfiltered tokens so
        # that e.g. a course literally titled "Introduction" still keys.
        tokens = _WS_RE.sub(" ", text).split()
    return "-".join(sorted(tokens))


def slugify(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _NONSLUG_RE.sub("-", text).strip("-")


def url_slug(url: Optional[str]) -> Optional[str]:
    """Extract the stable platform key from a course URL.

    Coursera exposes /learn/<slug> and /specializations/<slug>; edX exposes
    /course/<slug>. The slug is the only genuinely stable identifier in
    these datasets -- row index is not, and titles drift between snapshots.
    """
    if url is None:
        return None
    raw = str(url).strip()
    if not raw:
        return None
    try:
        path = urlsplit(raw).path
    except ValueError:
        return None
    parts = [p for p in path.split("/") if p]
    if not parts:
        return None
    for marker in ("learn", "specializations", "professional-certificates", "course"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1].casefold()
    return parts[-1].casefold()


class UnionFind:
    """Disjoint-set with path compression, used to close merge decisions.

    Pairwise "these two are the same" decisions are not transitive on their
    own; union-find makes the resulting grouping consistent regardless of the
    order the pairs were evaluated in, which is what lets the dedup stage be
    order-independent.
    """

    def __init__(self, items: Iterable[str] = ()) -> None:
        self._parent: Dict[str, str] = {}
        for item in items:
            self.add(item)

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression, iterative to stay safe on long chains.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        # Deterministic tie-break: the lexicographically smaller id wins, so
        # the grouping does not depend on the order unions were applied.
        root, child = (ra, rb) if ra < rb else (rb, ra)
        self._parent[child] = root
        return root

    def groups(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), []).append(item)
        for members in out.values():
            members.sort()
        return out
