"""External signal adapters for congressional actor augmentation.

Collects economic, demographic and relational indicators described in the
simulation brief:

1. Campaign finance features via OpenFEC/ProPublica-style totals.
2. Congressional district demographics plus 2024 presidential margin.
3. Cosponsorship and committee overlap matrices.

Each method keeps a small JSON cache so that repeated runs reuse prior results
while still allowing researchers to plug in real API keys later.
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

import config

logger = logging.getLogger(__name__)


class ExternalSignalCollector:
    """Aggregates supplemental data streams for CongressMan preparation."""

    CACHE_FILE = "external_signals.json"

    def __init__(
        self,
        cache_dir: str = config.EXTERNAL_SIGNAL_CACHE_DIR,
        use_cache: bool = True,
        finance_provider: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
        committee_provider: Optional[Callable[[List[int]], Optional[Dict[int, List[str]]]]] = None,
    ):
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(self.cache_dir, self.CACHE_FILE)
        self.use_cache = use_cache
        self._ensure_cache_dir()
        self.actor_cache: Dict[str, Dict] = self._load_cache()
        self.session = requests.Session() if requests is not None else None
        # Optional live data providers that can replace synthetic defaults
        self.finance_provider = finance_provider
        self.committee_provider = committee_provider

    def _ensure_cache_dir(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_cache(self) -> Dict[str, Dict]:
        if not self.use_cache or not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as error:  # pragma: no cover - best effort
            logger.debug("Failed to load external signal cache: %s", error)
            return {}

    def _save_cache(self) -> None:
        if not self.use_cache:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as handle:
                json.dump(self.actor_cache, handle, indent=2)
        except Exception as error:  # pragma: no cover - best effort
            logger.debug("Failed to persist external signal cache: %s", error)

    def collect_signals(self, actor_ids: List[int]) -> Dict[str, object]:
        """Return finance, district and relationship signals for each actor."""
        actor_signals: Dict[str, Dict] = {}
        for leg_id in actor_ids:
            key = str(leg_id)
            if key in self.actor_cache:
                actor_signals[key] = self.actor_cache[key]
            else:
                actor_signals[key] = self._build_actor_signal(leg_id)
                self.actor_cache[key] = actor_signals[key]
        self._save_cache()

        committee_roster = self._assign_committees(actor_ids)
        cosponsorship_matrix = self._build_cosponsorship_matrix(actor_ids)
        committee_matrix = self._build_committee_matrix(actor_ids, committee_roster)

        for idx, leg_id in enumerate(actor_ids):
            key = str(leg_id)
            relationships = actor_signals[key].setdefault("relationships", {})
            relationships["cosponsor_strength"] = float(np.mean(cosponsorship_matrix[idx]))
            relationships["committee_overlap"] = float(np.mean(committee_matrix[idx]))
            relationships["committee_memberships"] = committee_roster.get(leg_id, [])

        return {
            "actors": actor_signals,
            "cosponsorship_matrix": cosponsorship_matrix,
            "committee_matrix": committee_matrix,
        }

    # ------------------------------------------------------------------
    # Signal builders
    # ------------------------------------------------------------------

    def _build_actor_signal(self, leg_id: int) -> Dict[str, Dict]:
        finance = self._build_finance_profile(leg_id)
        district = self._build_district_profile(leg_id)
        relationships: Dict[str, object] = {
            "community_affinity": finance.get("pac_share", 0.0)
        }
        return {
            "finance": finance,
            "district": district,
            "relationships": relationships,
        }

    def _build_finance_profile(self, leg_id: int) -> Dict[str, object]:
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
        months_to_election = self._months_to_next_election()
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

    # ------------------------------------------------------------------
    # Relationship builders
    # ------------------------------------------------------------------

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

    def _months_to_next_election(self) -> int:
        today = date.today()
        target = date(2026, 11, 4)
        total_months = (target.year - today.year) * 12 + (target.month - today.month)
        return max(1, total_months)

    # ------------------------------------------------------------------
    # API integration helpers (placeholders)
    # ------------------------------------------------------------------

    def fetch_fec_summary(self, candidate_id: str) -> Optional[Dict[str, object]]:
        """Placeholder that knows how to call OpenFEC or ProPublica given API keys."""
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
        """Placeholder ACS fetcher for future integration."""
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
        cache_path: str = config.EXTERNAL_METRICS_CACHE_PATH,
        max_age_days: int = config.EXTERNAL_METRICS_MAX_AGE_DAYS,
        vix_url: str = config.EXTERNAL_METRICS_VIX_URL,
        approval_url: str = config.EXTERNAL_METRICS_APPROVAL_URL,
        polarization_url: str = config.EXTERNAL_METRICS_POLARIZATION_URL,
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
                        or row.get("mean_dw_nominate")
                        or row.get("coord1D")
                        or row.get("mean")
                    )
                    mean_val = float(mean_raw)
                except Exception:
                    continue
                if congress <= 0:
                    continue
                congress_means.setdefault(congress, {})[party_code] = mean_val

            if not congress_means:
                return None

            latest_congress = max(congress_means.keys())
            party_means = congress_means.get(latest_congress, {})
            if 100 not in party_means or 200 not in party_means:
                return None

            spread = abs(party_means[100] - party_means[200])
            # DW-NOMINATE spreads rarely exceed ~2; normalize to [0,1]
            return float(np.clip(spread / 2.0, 0.0, 1.0))
        except Exception as error:
            logger.debug("Failed to fetch polarization: %s", error)
            return None


class OpenFECFinanceProvider:
    """Optional OpenFEC-backed finance provider (requires FEC_API_KEY).

    Provide a candidate lookup callable mapping internal leg_id -> FEC candidate ID.
    The provider is callable so it can be passed directly into ExternalSignalCollector.
    """

    def __init__(
        self,
        candidate_lookup: Callable[[int], Optional[str]],
        cycle: int = 2024,
        session: Optional[Any] = None,
    ) -> None:
        self.candidate_lookup = candidate_lookup
        self.cycle = cycle
        self.session = session or (requests.Session() if requests is not None else None)
        self.api_key = os.environ.get("FEC_API_KEY")

    def __call__(self, leg_id: int) -> Optional[Dict[str, Any]]:
        if not self.session or not self.api_key or not self.candidate_lookup:
            return None
        candidate_id = self.candidate_lookup(leg_id)
        if not candidate_id:
            return None
        try:
            url = f"https://api.open.fec.gov/v1/candidate/{candidate_id}/totals/"
            params = {"api_key": self.api_key, "cycle": self.cycle, "election_full": True}
            response = self.session.get(url, params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []
            if not results:
                return None
            row = results[0]
            total_raised = float(row.get("receipts", 0.0))
            total_spent = float(row.get("disbursements", 0.0))
            pac_receipts = float(row.get("pac_receipts", 0.0))
            individual = float(row.get("individual_contributions", 0.0))
            total = max(total_raised, 1e-6)
            pac_share = pac_receipts / total
            small_donor_share = max(0.0, 1.0 - pac_share)
            return {
                "cycle": self.cycle,
                "total_raised": round(total_raised, 2),
                "total_spent": round(total_spent, 2),
                "pac_share": round(pac_share, 3),
                "small_donor_share": round(small_donor_share, 3),
                "pac_to_small_ratio": round((pac_share + 1e-3) / (small_donor_share + 1e-3), 3),
                "individual_contributions": round(individual, 2),
            }
        except Exception as error:
            logger.debug("OpenFEC finance fetch failed for %s: %s", leg_id, error)
            return None


class ProPublicaCommitteeProvider:
    """Optional ProPublica provider for committee memberships (requires PROPUBLICA_API_KEY).

    Provide a member lookup callable mapping internal leg_id -> ProPublica member id.
    The provider returns a roster dict compatible with ExternalSignalCollector._assign_committees.
    """

    def __init__(
        self,
        member_lookup: Callable[[int], Optional[str]],
        session: Optional[Any] = None,
    ) -> None:
        self.member_lookup = member_lookup
        self.session = session or (requests.Session() if requests is not None else None)
        self.api_key = os.environ.get("PROPUBLICA_API_KEY")

    def __call__(self, actor_ids: List[int]) -> Optional[Dict[int, List[str]]]:
        if not self.session or not self.api_key or not self.member_lookup:
            return None
        roster: Dict[int, List[str]] = {}
        headers = {"X-API-Key": self.api_key}
        for leg_id in actor_ids:
            member_id = self.member_lookup(leg_id)
            if not member_id:
                continue
            try:
                url = f"https://api.propublica.org/congress/v1/members/{member_id}/committees.json"
                response = self.session.get(url, headers=headers, timeout=8)
                response.raise_for_status()
                payload = response.json()
                results = payload.get("results") or []
                if not results:
                    continue
                committees_raw = results[0].get("committees") or []
                roster[leg_id] = [c.get("name") for c in committees_raw if c.get("name")]
            except Exception as error:
                logger.debug("Committee fetch failed for %s: %s", leg_id, error)
        return roster if roster else None


__all__ = ["ExternalSignalCollector"]
