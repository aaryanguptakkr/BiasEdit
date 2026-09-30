"""Infini-gram API client with SQLite caching at full precision."""

import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_olmo-mix-1124_llama"
FULL_PRECISION_MCF = 500000


class InfinigramClient:
    """Client for Infini-gram n-gram and co-occurrence frequency queries.

    Rationale for on-disk SQLite caching:
      1. Query Deduplication: Target words ('female', 'he', 'nurse') and attribute words
         ('strong', 'weak', 'lazy') repeat heavily across the 4,597 StereoSet pairs.
         Caching reduces total queries from ~23,000 down to ~7,850 (~66% reduction).
      2. Resumability & Rate Limits: The public API throttles at ~4-5 req/s. Disk caching
         ensures network timeouts, rate limits, or job restarts resume instantly from where
         they stopped rather than repeating a 25-minute run from scratch.
      3. Fast Re-analysis: Allows downstream correlation and metric re-runs to execute
         locally in milliseconds without re-fetching across the internet.
    """

    def __init__(
        self,
        index: str = DEFAULT_INDEX,
        cache_path: str = "outputs/cooccurrence/cache.db",
        timeout: float = 8.0,
    ) -> None:
        self.index = index
        self.timeout = timeout
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create the cache table if it does not already exist."""
        with self._lock, sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    index_name TEXT,
                    query TEXT,
                    max_diff INTEGER,
                    mcf INTEGER,
                    count INTEGER,
                    is_approx INTEGER,
                    latency REAL,
                    PRIMARY KEY (index_name, query, max_diff, mcf)
                )
                """
            )

    def _get_cached(self, query: str, max_diff: int, mcf: int) -> tuple[int, bool, float] | None:
        """Retrieve a cached query result if available."""
        with self._lock, sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                "SELECT count, is_approx, latency FROM query_cache WHERE index_name=? AND query=? AND max_diff=? AND mcf=?",
                (self.index, query, max_diff, mcf),
            ).fetchone()
        return (row[0], bool(row[1]), row[2]) if row else None

    def _save_cache(self, query: str, max_diff: int, mcf: int, count: int, approx: bool, latency: float) -> None:
        """Store a successful query result into the local cache."""
        with self._lock, sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.index, query, max_diff, mcf, count, int(approx), latency),
            )

    def _post(self, payload: dict[str, Any], max_retries: int = 8) -> dict[str, Any]:
        """Send a JSON POST request to the Infini-gram API with backoff on rate limits."""
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) BiasResearch/1.0",
        }
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(API_URL, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in (403, 429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    wait_time = min(60.0, 2.0 * (2**attempt))
                    time.sleep(wait_time)
                    continue
                raise

    def count(self, term: str) -> int:
        """Get the exact unigram count for a single term."""
        cached = self._get_cached(term, 0, 0)
        if cached:
            return cached[0]
        payload = {"index": self.index, "query_type": "count", "query": term}
        res = self._post(payload)
        count = int(res.get("count", 0))
        self._save_cache(term, 0, 0, count, bool(res.get("approx", False)), float(res.get("latency", 0.0)))
        return count

    def count_cooccurrence(
        self, term1: str, term2: str, max_diff_tokens: int = 50
    ) -> tuple[int, bool, float]:
        """Query co-occurrence with full precision (500,000 clause threshold)."""
        query = f"{term1} AND {term2}"
        cached = self._get_cached(query, max_diff_tokens, FULL_PRECISION_MCF)
        if cached:
            return cached

        payload = {
            "index": self.index,
            "query_type": "count",
            "query": query,
            "max_diff_tokens": max_diff_tokens,
            "max_clause_freq": FULL_PRECISION_MCF,
        }
        res = self._post(payload)
        cnt = int(res.get("count", 0))
        app = bool(res.get("approx", False))
        lat = float(res.get("latency", 0.0))
        self._save_cache(query, max_diff_tokens, FULL_PRECISION_MCF, cnt, app, lat)
        return cnt, app, lat
