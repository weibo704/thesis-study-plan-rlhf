"""Cached, deterministic OpenAI access.

Every call is keyed by sha256 over (model, payload) and persisted under
`data/catalog/llm_cache/`. A rebuild replays the cache and needs no API
access, which is what makes the taxonomy reproducible by a reader --
docs/ingestion-plan.md section 3, "Determinism requirements".

Two rules are enforced here rather than left to callers:

  * temperature is a DECLARED property of each role (config/llm.yaml), set
    once and cached with the response -- never varied within a role
  * there is still no retry-at-higher-temperature. Retries are
    transport-level only, with identical parameters. Resampling a FAILED
    call until it complies destroys reproducibility for exactly the hard
    cases, and remains banned.

The cache key includes temperature, so a replay is exact even for a warm
role: the first call is sampled, every later call with the same prompt
replays that sample.

The API key is read from the environment and is never logged or written
into any artefact.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from . import paths

CACHE_DIR = paths.CATALOG / "llm_cache"
DEFAULT_TEMPERATURE = 0.0


def role_temperature(role: Optional[str]) -> float:
    """Declared temperature for a role, from config/llm.yaml."""
    if not role:
        return DEFAULT_TEMPERATURE
    return float(config().get("temperature", {}).get(role, DEFAULT_TEMPERATURE))


def config() -> Dict[str, Any]:
    with open(paths.CONFIG / "llm.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _cache_key(kind: str, model: str, payload: Any) -> str:
    blob = json.dumps({"kind": kind, "model": model, "payload": payload},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    # Shard by prefix so the directory stays navigable at tens of thousands
    # of entries.
    return CACHE_DIR / key[:2] / f"{key}.json"


def cache_get(key: str) -> Optional[Any]:
    path = _cache_path(key)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["response"]
    return None


def cache_put(key: str, kind: str, model: str, response: Any) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"kind": kind, "model": model, "response": response},
                  fh, ensure_ascii=False)
    tmp.replace(path)


def cache_stats() -> Dict[str, int]:
    if not CACHE_DIR.exists():
        return {"entries": 0}
    return {"entries": sum(1 for _ in CACHE_DIR.rglob("*.json"))}


class LLMUnavailable(RuntimeError):
    pass


def _client():
    key = paths.llm_api_key()
    if not key:
        raise LLMUnavailable(
            "OPENAI_API_KEY is not set. Create .env (see .env.example)."
        )
    from openai import OpenAI
    return OpenAI(api_key=key, timeout=config()["limits"]["request_timeout_seconds"])


class Budget:
    """Hard call ceiling. Aborts rather than silently overrunning cost."""

    def __init__(self, limit: int, label: str) -> None:
        self.limit, self.label, self.used = limit, label, 0

    def spend(self, n: int = 1) -> None:
        self.used += n
        if self.used > self.limit:
            raise RuntimeError(
                f"{self.label} budget exceeded ({self.used} > {self.limit}). "
                f"Raise limits in config/llm.yaml if this is intended."
            )


def chat_json(
    messages: Sequence[Dict[str, str]],
    model: str,
    budget: Optional[Budget] = None,
    role: Optional[str] = None,
) -> Optional[Any]:
    """One JSON-returning chat completion, cached. None on failure.

    `role` selects the declared temperature. The cache key includes it, so
    a warm role still replays exactly.
    """
    temperature = role_temperature(role)
    payload = {"messages": list(messages), "temperature": temperature,
               "response_format": "json_object"}
    key = _cache_key("chat", model, payload)
    hit = cache_get(key)
    if hit is not None:
        return hit

    if budget is not None:
        budget.spend()
    client = _client()
    last_error: Optional[Exception] = None
    for attempt in range(config()["limits"]["max_retries"]):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,   # declared per role; never raised on retry
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            cache_put(key, "chat", model, parsed)
            return parsed
        except json.JSONDecodeError as exc:
            # A malformed response is a real failure, not a transport blip.
            # Do NOT resample at a higher temperature; log and move on.
            last_error = exc
            break
        except Exception as exc:  # noqa: BLE001 - transport/rate limit
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"[llm] chat failed ({type(last_error).__name__}); recorded as a miss")
    return None


def embed(texts: Sequence[str], model: str,
          budget: Optional[Budget] = None) -> np.ndarray:
    """Embed texts, caching per individual text so partial reuse works."""
    texts = [t if t else " " for t in texts]
    keys = [_cache_key("embed", model, t) for t in texts]
    vectors: List[Optional[List[float]]] = [cache_get(k) for k in keys]

    missing = [i for i, v in enumerate(vectors) if v is None]
    if missing:
        client = _client()
        batch_size = config()["limits"]["embed_batch"]
        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            if budget is not None:
                budget.spend(len(chunk))
            resp = client.embeddings.create(
                model=model, input=[texts[i] for i in chunk]
            )
            for item, idx in zip(resp.data, chunk):
                vectors[idx] = item.embedding
                cache_put(keys[idx], "embed", model, item.embedding)

    arr = np.asarray(vectors, dtype=np.float32)
    # L2-normalise so a dot product is cosine similarity.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-9, None)
