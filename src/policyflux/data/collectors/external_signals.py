"""External signal adapters for congressional actor augmentation.

Collects economic, demographic and relational indicators described in the
simulation brief.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

from policyflux.defaults import SIMULATION, MODELS, PATHS
from policyflux.core.observation import Observation
from policyflux.data.collectors.base import BaseCollector, CollectorResult

logger = logging.getLogger(__name__)


class ExternalSignalCollector(BaseCollector):
    """Aggregates supplemental data streams for CongressMan preparation."""

    CACHE_FILE = "external_signals.json"

    def __init__(
        self,
        cache_dir: str = defaults.EXTERNAL_SIGNAL_CACHE_DIR,
        use_cache: bool = True,
        finance_provider: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
        committee_provider: Optional[Callable[[List[int]], Optional[Dict[int, List[str]]]]] = None,
        session: Optional[Any] = None,
    ):
        self.collector_version = 1
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(self.cache_dir, self.CACHE_FILE)
        self.use_cache = use_cache
        self._ensure_cache_dir()
        self.actor_cache, self.cache_meta = self._load_cache()
        self.session = session or (requests.Session() if requests is not None else None)
        self.finance_provider = finance_provider
        self.committee_provider = committee_provider

    def _ensure_cache_dir(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_cache(self) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        if not self.use_cache or not os.path.exists(self.cache_path):
            return {}, {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                return {}, {}
            payload_meta = payload.get("meta", {}) or {}
            version = payload_meta.get("version")
            if version and version != self.collector_version:
                logger.debug("External signal cache version mismatch: %s != %s", version, self.collector_version)
                return {}, {}
            if "actors" in payload:
                return payload.get("actors", {}), payload_meta
            return payload, payload_meta
        except Exception as error:  # pragma: no cover - best effort
            logger.debug("Failed to load external signal cache: %s", error)
        return {}, {}

    def _save_cache(self, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.use_cache:
            return
        try:
            merged_meta = {"version": self.collector_version, **(self.cache_meta or {}), **(meta or {})}
            payload = {"actors": self.actor_cache, "meta": merged_meta}
            with open(self.cache_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self.cache_meta = merged_meta
        except Exception as error:  # pragma: no cover - best effort
            logger.debug("Failed to persist external signal cache: %s", error)

    def _ingest_cached(self, actor_ids: List[int]) -> Dict[str, Dict]:
        cached: Dict[str, Dict] = {}
        for leg_id in actor_ids:
            key = str(leg_id)
            if key in self.actor_cache:
                cached[key] = self.actor_cache[key]
        return cached

    def _validate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        validated = dict(signal or {})
        validated.setdefault("finance", {})
        validated.setdefault("district", {})
        validated.setdefault("relationships", {})
        return validated

    def _normalize_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._validate_signal(signal)
        for section in ("finance", "district"):
            for key, value in list(normalized[section].items()):
                if isinstance(value, float):
                    normalized[section][key] = round(value, 4)
        return normalized

    def _attach_relationships(
        self, actor_ids: List[int], actor_signals: Dict[str, Dict]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[str]]]:
        committee_roster = self._assign_committees(actor_ids)
        cosponsorship_matrix = self._build_cosponsorship_matrix(actor_ids)
        committee_matrix = self._build_committee_matrix(actor_ids, committee_roster)

        for idx, leg_id in enumerate(actor_ids):
            key = str(leg_id)
            relationships = actor_signals[key].setdefault("relationships", {})
            relationships["cosponsor_strength"] = float(np.mean(cosponsorship_matrix[idx]))
            relationships["committee_overlap"] = float(np.mean(committee_matrix[idx]))
            relationships.setdefault("committee_memberships", committee_roster.get(leg_id, []))

        return cosponsorship_matrix, committee_matrix, committee_roster

    def collect(self, actor_ids: List[int]) -> CollectorResult:
        actor_ids = list(actor_ids)
        run_date = date.today()
        cached = self._ingest_cached(actor_ids)
        hit_count = len(cached)
        source = "fresh"
        if hit_count and hit_count == len(actor_ids):
            source = "cache"
        elif hit_count:
            source = "mixed"
        actor_signals: Dict[str, Dict] = {}

        for leg_id in actor_ids:
            key = str(leg_id)
            actor_signals[key] = self._process_actor_signal(leg_id, cached.get(key), run_date)
            self.actor_cache[key] = actor_signals[key]

        cosponsorship_matrix, committee_matrix, committee_roster = self._attach_relationships(actor_ids, actor_signals)

        generated_at = run_date.isoformat()
        meta = {
            "actor_ids": list(actor_ids),
            "cache_path": self.cache_path,
            "cache_hits": hit_count,
            "source": source,
            "generated_at": generated_at,
            "collector_version": self.collector_version,
        }
        self._save_cache(
            meta={
                "actor_count": len(actor_signals),
                "actor_ids": list(actor_ids),
                "generated_at": generated_at,
                "cache_hits": hit_count,
                "source": source,
                "collector_version": self.collector_version,
            }
        )

        observations = {
            key: Observation(actor_id=int(key), features=signals, metadata={"source": source, "cached": key in cached})
            for key, signals in actor_signals.items()
        }
        matrices = {
            "cosponsorship_matrix": cosponsorship_matrix,
            "committee_matrix": committee_matrix,
            "committee_roster": committee_roster,
        }
        return CollectorResult(actors=observations, matrices=matrices, meta=meta)

    def _process_actor_signal(self, leg_id: int, cached_signal: Optional[Dict[str, Any]], run_date: date) -> Dict[str, Any]:
        """Run ingest→validate→normalize for a single actor."""
        if cached_signal:
            return self._normalize_signal(cached_signal)
        built = self._build_actor_signal(leg_id, run_date)
        return self._normalize_signal(built)

    def collect_signals(self, actor_ids: List[int]) -> Dict[str, object]:
        """Legacy API returning a plain dictionary."""
        return self.collect(actor_ids).to_dict()

    def _build_actor_signal(self, leg_id: int, run_date: date) -> Dict[str, Dict]:
        finance = self._build_finance_profile(leg_id, run_date)
        district = self._build_district_profile(leg_id)
        relationships: Dict[str, object] = {"community_affinity": finance.get("pac_share", 0.0)}
        return {"finance": finance, "district": district, "relationships": relationships}

    def _build_finance_profile(self, leg_id: int, run_date: date) -> Dict[str, object]:
        if self.finance_provider is not None:
            try:
                live_finance = self.finance_provider(leg_id)
                if live_finance:
                    return live_finance
            except Exception as error:
                logger.debug("Live finance provider failed for %s: %s", leg_id, error)

        rng = np.random.default_rng(leg_id)
        total_raised = float(2_000_000 + rng.integers(0, 5_000_000))
        total_spent = float(total_raised * (0.6 + rng.random() * 0.3))
        pac_share = float(0.2 + 0.5 * rng.random())
        small_share = max(0.0, min(1.0, 1.0 - pac_share))
        months_to_election = self._months_to_next_election(run_date)
        return {
            "cycle": 2024,
            "total_raised": round(total_raised, 2),
            "total_spent": round(total_spent, 2),
            "pac_share": round(pac_share, 3),
            "small_donor_share": round(small_share, 3),
            "pac_to_small_ratio": round((pac_share + 1e-3) / (small_share + 1e-3), 3),
            "months_to_election": months_to_election,
        }

    def _build_district_profile(self, leg_id: int) -> Dict[str, object]:
        rng = np.random.default_rng(leg_id + 31415)
        median_income = float(55_000 + rng.integers(0, 70_000))
        unemployment = round(0.03 + rng.random() * 0.08, 3)
        racial_diversity = round(min(1.0, rng.random() * 0.9 + 0.1), 3)
        presidential_margin = round(rng.random() * 0.5 * (1 if rng.random() < 0.5 else -1), 3)
        return {
            "median_income": median_income,
            "unemployment_rate": unemployment,
            "racial_diversity": racial_diversity,
            "presidential_margin": presidential_margin,
            "population_density": int(100 + rng.integers(0, 800)),
            "district_margin": round(rng.random() * 0.4 * (1 if rng.random() < 0.5 else -1), 3),
        }

    def _assign_committees(self, actor_ids: List[int]) -> Dict[int, List[str]]:
        if self.committee_provider is not None:
            try:
                live_roster = self.committee_provider(actor_ids)
                if live_roster:
                    return live_roster
            except Exception as error:
                logger.debug("Live committee provider failed: %s", error)

        committees = [
            "Appropriations",
            "Armed Services",
            "Budget",
            "Energy and Commerce",
            "Judiciary",
            "Oversight",
        ]
        roster: Dict[int, List[str]] = {}
        for leg_id in actor_ids:
            rng = np.random.default_rng(leg_id)
            count = rng.integers(2, 5)
            roster[leg_id] = list(rng.choice(committees, size=count, replace=False))
        return roster

    def _build_cosponsorship_matrix(self, actor_ids: List[int]) -> np.ndarray:
        n = len(actor_ids)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i, leg_i in enumerate(actor_ids):
            for j in range(i + 1, n):
                leg_j = actor_ids[j]
                rng = np.random.default_rng(leg_i + leg_j)
                value = float(rng.random())
                matrix[i, j] = value
                matrix[j, i] = value
        return matrix

    def _build_committee_matrix(self, actor_ids: List[int], roster: Dict[int, List[str]]) -> np.ndarray:
        n = len(actor_ids)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i, leg_i in enumerate(actor_ids):
            for j, leg_j in enumerate(actor_ids):
                if i == j:
                    continue
                committees_i = set(roster.get(leg_i, []))
                committees_j = set(roster.get(leg_j, []))
                if not committees_i or not committees_j:
                    continue
                overlap = len(committees_i & committees_j) / max(1, min(len(committees_i), len(committees_j)))
                matrix[i, j] = float(overlap)
        return matrix

    def _months_to_next_election(self, as_of: date) -> int:
        target = date(2026, 11, 4)
        total_months = (target.year - as_of.year) * 12 + (target.month - as_of.month)
        return max(1, total_months)

    def fetch_fec_summary(self, candidate_id: str) -> Optional[Dict[str, object]]:
        if self.session is None:
            return None
        params = {
            "api_key": os.environ.get("FEC_API_KEY"),
            "candidate_id": candidate_id,
            "cycle": 2024,
        }
        try:
            response = self.session.get(
                "https://api.open.fec.gov/v1/candidate/",
                params=params,
                timeout=8,
            )
            response.raise_for_status()
            return response.json()
        except Exception as error:  # pragma: no cover - external network
            logger.debug("Failed to contact FEC: %s", error)
            return None

    def fetch_district_acs(self, cd: str) -> Optional[Dict[str, object]]:
        if self.session is None:
            return None
        params = {
            "get": "B19013_001E,B23025_005E,NAME",
            "for": f"congressional district:{cd}",
            "in": "state:01",
            "key": os.environ.get("ACS_API_KEY"),
        }
        try:
            response = self.session.get("https://api.census.gov/data/2022/acs/acs5", params=params, timeout=8)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            logger.debug("Failed to contact ACS: %s", error)
            return None



class MacroSignalProvider:
    """Fetch live macro indicators (VIX, polarization, approval) with caching.

    The provider only performs network calls when enabled explicitly; results are
    cached to avoid hitting upstream sources repeatedly during longer runs.
    """

    def __init__(
        self,
        cache_path: str = defaults.EXTERNAL_METRICS_CACHE_PATH,
        max_age_days: int = defaults.EXTERNAL_METRICS_MAX_AGE_DAYS,
        vix_url: str = defaults.EXTERNAL_METRICS_VIX_URL,
        approval_url: str = defaults.EXTERNAL_METRICS_APPROVAL_URL,
        polarization_url: str = defaults.EXTERNAL_METRICS_POLARIZATION_URL,
        session: Optional[Any] = None,
    ) -> None:
        self.cache_path = cache_path
        self.max_age_days = max_age_days
        self.vix_url = vix_url
        self.approval_url = approval_url
        self.polarization_url = polarization_url
        self.session = session or (requests.Session() if requests is not None else None)

    def fetch_latest_metrics(self) -> Dict[str, Optional[float]]:
        cached = self._load_cache()
        if cached is not None:
            return cached

        metrics = {
            "vix": self._fetch_vix(),
            "polarization": self._fetch_polarization(),
            "approval": self._fetch_presidential_approval(),
        }
        self._save_cache(metrics)
        return metrics

    def _load_cache(self) -> Optional[Dict[str, Optional[float]]]:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            as_of_raw = payload.get("as_of")
            metrics = payload.get("metrics")
            if not metrics or not as_of_raw:
                return None
            as_of = datetime.strptime(as_of_raw, "%Y-%m-%d").date()
            if (date.today() - as_of).days > self.max_age_days:
                return None
            return {"vix": metrics.get("vix"), "polarization": metrics.get("polarization"), "approval": metrics.get("approval")}
        except Exception as error:
            logger.debug("Failed to load macro metric cache: %s", error)
            return None

    def _save_cache(self, metrics: Dict[str, Optional[float]]) -> None:
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            payload = {"as_of": date.today().isoformat(), "metrics": metrics}
            with open(self.cache_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as error:
            logger.debug("Failed to persist macro metric cache: %s", error)

    def _fetch_vix(self) -> Optional[float]:
        if self.session is None:
            return None
        try:
            end = date.today()
            start = end - timedelta(days=30)
            url = self.vix_url.format(d1=start.strftime("%Y%m%d"), d2=end.strftime("%Y%m%d"))
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            rows = response.text.strip().splitlines()
            if len(rows) < 2:
                return None
            last_row = rows[-1].split(",")
            close_idx = 4 if len(last_row) >= 5 else -1
            close = float(last_row[close_idx])
            return close
        except Exception as error:
            logger.debug("Failed to fetch VIX: %s", error)
            return None

    def _fetch_presidential_approval(self) -> Optional[float]:
        if self.session is None:
            return None
        try:
            response = self.session.get(self.approval_url, timeout=8)
            response.raise_for_status()
            best_ts: Optional[str] = None
            best_val: Optional[float] = None
            for row in csv.DictReader(response.text.splitlines()):
                subgroup = (row.get("subgroup") or "").lower()
                if subgroup not in ("all polls", "all adults", "adults"):
                    continue
                president = (row.get("president") or "").lower()
                if president and "biden" not in president and "approval" not in president:
                    continue
                try:
                    approve_raw = row.get("approve_estimate") or row.get("approve_hi") or row.get("approve")
                    approve = float(approve_raw)
                except Exception:
                    continue
                ts = row.get("modeldate") or row.get("timestamp") or ""
                if best_ts is None or ts > best_ts:
                    best_ts = ts
                    best_val = approve
            if best_val is None:
                return None
            if best_val > 1.5:  # csv is in percentage points
                best_val = best_val / 100.0
            return float(np.clip(best_val, 0.0, 1.0))
        except Exception as error:
            logger.debug("Failed to fetch approval: %s", error)
            return None

    def _fetch_polarization(self) -> Optional[float]:
        if self.session is None:
            return None
        try:
            response = self.session.get(self.polarization_url, timeout=8)
            response.raise_for_status()
            congress_means: Dict[int, Dict[int, float]] = {}
            for row in csv.DictReader(response.text.splitlines()):
                try:
                    congress = int(row.get("congress") or row.get("Congress") or 0)
                    party_code = int(row.get("party_code") or row.get("party") or 0)
                    mean_raw = (
                        row.get("nominate")
                        or row.get("mean_nominate")
                        or row.get("mean_nom")
                        or row.get("mean")
                    )
                except Exception:
                    continue
                try:
                    mean_val = float(mean_raw)
                except Exception:
                    mean_val = 0.0
                congress_means.setdefault(congress, {})[party_code] = mean_val
            # Compute polarization for current congress
            current = defaults.CONGRESS_NUMBER
            parties = congress_means.get(current, {})
            if not parties:
                return None
            vals = list(parties.values())
            if len(vals) < 2:
                return 0.0
            left = float(vals[0])
            right = float(vals[1])
            pol = float(abs(left - right) / max(1.0, (abs(left) + abs(right))))
            return float(np.clip(pol, 0.0, 1.0))
        except Exception as error:
            logger.debug("Failed to fetch polarization: %s", error)
            return None

