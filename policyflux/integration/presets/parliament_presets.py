"""Ready-made multi-chamber parliament models for major legislative systems.

Each factory function returns a fully configured
:class:`~policyflux.toolbox.parliament_models.MultiChamberParliamentModel`
with one or two :class:`~policyflux.toolbox.congress_model.SequentialCongressModel`
chambers populated with :class:`~policyflux.toolbox.actor_models.SequentialVoter`
members using basic :class:`~policyflux.layers.ideal_point.IdealPointLayer` voting.

Supported systems
-----------------
- :func:`create_uk_parliament`        – Westminster bicameral (Commons + Lords)
- :func:`create_us_congress`          – US-style symmetric bicameral (House + Senate)
- :func:`create_german_parliament`    – Bundestag + Bundesrat (consent/non-consent)
- :func:`create_french_parliament`    – Assemblée Nationale + Sénat (navette)
- :func:`create_italian_parliament`   – Camera dei Deputati + Senato (perfect bicameralism)
- :func:`create_polish_parliament`    – Sejm + Senat (override by lower majority)
- :func:`create_swedish_parliament`   – Riksdag (unicameral)
- :func:`create_spanish_parliament`   – Congreso + Senado (weak upper)
- :func:`create_australian_parliament`– House of Representatives + Senate (symmetric)
- :func:`create_canadian_parliament`  – House of Commons + Senate (advisory upper)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.pf_typing import PolicySpace
from ..layers.ideal_point import IdealPointLayer
from ..pfrandom import random as pf_random
from .actor_models import SequentialVoter
from .congress_model import SequentialCongressModel
from .parliament_models import (
    ChamberConfig,
    ChamberRole,
    MultiChamberParliamentModel,
    PassageThreshold,
    UpperChamberPowers,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_chamber(
    n_members: int,
    policy_dim: int,
    member_prefix: str,
    **voter_kwargs: Any,
) -> SequentialCongressModel:
    """Create a :class:`SequentialCongressModel` populated with *n_members* voters.

    Each voter gets a random ideal point in *policy_dim* dimensions.
    """
    chamber = SequentialCongressModel()
    for i in range(1, n_members + 1):
        space = PolicySpace(policy_dim)
        space.set_position([pf_random() for _ in range(policy_dim)])
        status_quo = PolicySpace(policy_dim)
        status_quo.set_position([0.5] * policy_dim)
        ideal_point_layer = IdealPointLayer(space=space, status_quo=status_quo)

        voter = SequentialVoter(
            id=None,
            name=f"{member_prefix}-{i}",
            layers=[ideal_point_layer],
            **voter_kwargs,
        )
        chamber.add_congressman(voter)
    chamber.compile()
    return chamber


# ---------------------------------------------------------------------------
# Parliament configurations as dataclasses (for type-safe overrides)
# ---------------------------------------------------------------------------


@dataclass
class ParliamentPresetConfig:
    """Parameters for tuning a parliament preset."""

    policy_dim: int = 2
    """Number of policy dimensions for member ideal points."""

    lower_house_size: int | None = None
    """Override the default lower-house membership size."""

    upper_house_size: int | None = None
    """Override the default upper-house membership size."""

    parliament_name: str | None = None
    """Override the default parliament name."""


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------


def create_uk_parliament(
    config: ParliamentPresetConfig | None = None,
) -> MultiChamberParliamentModel:
    """Westminster bicameral parliament.

    - **House of Commons** (650 seats) – simple majority, full legislative power.
    - **House of Lords** (~800 members) – suspensive veto only; Commons can override
      after one round by re-passing the bill (Parliament Acts 1911/1949).
    - Money bills are exempt from Lords interference (``budget_bill_exempt=True``).

    Constitutional reference:
        Parliament Acts 1911 and 1949 (United Kingdom).
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "UK Parliament"
    policy_dim = cfg.policy_dim
    commons_size = cfg.lower_house_size or 650
    lords_size = cfg.upper_house_size or 800

    commons = _make_chamber(commons_size, policy_dim, "MP")
    lords = _make_chamber(lords_size, policy_dim, "Lord")

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        commons,
        ChamberConfig(
            name="House of Commons",
            role=ChamberRole.LOWER,
            size=commons_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    parliament.add_chamber(
        lords,
        ChamberConfig(
            name="House of Lords",
            role=ChamberRole.UPPER,
            size=lords_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
            powers=UpperChamberPowers.SUSPENSIVE_VETO,
            max_ping_pong_rounds=1,  # Lords can only delay by one parliamentary session
            budget_bill_exempt=True,
        ),
    )
    return parliament


def create_us_congress(
    config: ParliamentPresetConfig | None = None,
) -> MultiChamberParliamentModel:
    """US-style symmetric bicameral congress.

    - **House of Representatives** (435 seats) – simple majority.
    - **Senate** (100 seats) – simple majority; full veto (both must pass identical text).
    - Constitutional reference: Article I, United States Constitution.

    Note: this model does *not* simulate the Senate filibuster (which requires
    60 votes for cloture). Override ``upper_house_size`` and threshold in the
    config, or post-hoc adjust the ``ChamberConfig.passage_threshold`` if needed.
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "US Congress"
    policy_dim = cfg.policy_dim
    house_size = cfg.lower_house_size or 435
    senate_size = cfg.upper_house_size or 100

    house = _make_chamber(house_size, policy_dim, "Rep")
    senate = _make_chamber(senate_size, policy_dim, "Senator")

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        house,
        ChamberConfig(
            name="House of Representatives",
            role=ChamberRole.LOWER,
            size=house_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    parliament.add_chamber(
        senate,
        ChamberConfig(
            name="Senate",
            role=ChamberRole.UPPER,
            size=senate_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
            powers=UpperChamberPowers.FULL_VETO,
        ),
    )
    return parliament


def create_german_parliament(
    config: ParliamentPresetConfig | None = None,
    *,
    consent_law: bool = True,
) -> MultiChamberParliamentModel:
    """German federal parliament (Bundestag + Bundesrat).

    - **Bundestag** (736 seats) – simple majority.
    - **Bundesrat** (69 votes representing Länder) –

      * *Consent laws* (``consent_law=True``): Bundesrat has a full veto.
        These are laws that affect the Länder directly (about 40–50 % of all laws).
      * *Non-consent laws* (``consent_law=False``): Bundesrat's veto can be
        overridden by the Bundestag with the same majority used to pass the law
        (simple majority → simple majority override; absolute majority needed
        for an absolute-majority bundesrat objection).

    Constitutional reference:
        Articles 77–78, Grundgesetz (Basic Law) of Germany.
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "Deutscher Bundestag/Bundesrat"
    policy_dim = cfg.policy_dim
    bundestag_size = cfg.lower_house_size or 736
    bundesrat_size = cfg.upper_house_size or 69

    bundestag = _make_chamber(bundestag_size, policy_dim, "MdB")
    bundesrat = _make_chamber(bundesrat_size, policy_dim, "BRat")

    powers = UpperChamberPowers.FULL_VETO if consent_law else UpperChamberPowers.OVERRIDE_BY_LOWER
    override_threshold = 0.5 if not consent_law else 0.5  # absolute majority of Bundestag

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        bundestag,
        ChamberConfig(
            name="Bundestag",
            role=ChamberRole.LOWER,
            size=bundestag_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    parliament.add_chamber(
        bundesrat,
        ChamberConfig(
            name="Bundesrat",
            role=ChamberRole.UPPER,
            size=bundesrat_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
            powers=powers,
            override_threshold=override_threshold,
        ),
    )

    law_type = "consent" if consent_law else "non-consent"
    parliament.name = f"{name} ({law_type} law)"
    return parliament


def create_french_parliament(
    config: ParliamentPresetConfig | None = None,
    *,
    max_navette_rounds: int = 2,
) -> MultiChamberParliamentModel:
    """French Parliament (Assemblée Nationale + Sénat), navette législative.

    - **Assemblée Nationale** (577 seats) – simple majority, can invoke *dernier mot*
      (last word) after *max_navette_rounds* rounds of navette.
    - **Sénat** (348 seats) – suspensive veto; AN always prevails in the end.
    - The ``max_navette_rounds`` parameter controls how many ping-pong exchanges
      occur before AN's position prevails.

    Constitutional reference:
        Article 45, Constitution of the Fifth French Republic (1958).
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "Parlement français"
    policy_dim = cfg.policy_dim
    an_size = cfg.lower_house_size or 577
    senat_size = cfg.upper_house_size or 348

    assemblee = _make_chamber(an_size, policy_dim, "Déput")
    senat = _make_chamber(senat_size, policy_dim, "Sénateur")

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        assemblee,
        ChamberConfig(
            name="Assemblée Nationale",
            role=ChamberRole.LOWER,
            size=an_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    parliament.add_chamber(
        senat,
        ChamberConfig(
            name="Sénat",
            role=ChamberRole.UPPER,
            size=senat_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
            powers=UpperChamberPowers.SUSPENSIVE_VETO,
            max_ping_pong_rounds=max_navette_rounds,
        ),
    )
    return parliament


def create_italian_parliament(
    config: ParliamentPresetConfig | None = None,
) -> MultiChamberParliamentModel:
    """Italian Parliament – *bicameralismo perfetto* (symmetric bicameralism).

    - **Camera dei Deputati** (400 seats) – simple majority.
    - **Senato della Repubblica** (200 seats + 6 life senators) –
      full veto; both chambers must pass *identical text*.

    This is one of the rare examples of a truly symmetric bicameral system:
    neither chamber is dominant.

    Constitutional reference:
        Articles 70–82, Costituzione della Repubblica Italiana (1948),
        as amended by constitutional law 2020 reducing seat numbers.
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "Parlamento italiano"
    policy_dim = cfg.policy_dim
    camera_size = cfg.lower_house_size or 400
    senato_size = cfg.upper_house_size or 206  # 200 elected + 6 life senators (approx)

    camera = _make_chamber(camera_size, policy_dim, "Dep")
    senato = _make_chamber(senato_size, policy_dim, "Sen")

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        camera,
        ChamberConfig(
            name="Camera dei Deputati",
            role=ChamberRole.LOWER,
            size=camera_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    parliament.add_chamber(
        senato,
        ChamberConfig(
            name="Senato della Repubblica",
            role=ChamberRole.UPPER,
            size=senato_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
            powers=UpperChamberPowers.FULL_VETO,
        ),
    )
    return parliament


def create_polish_parliament(
    config: ParliamentPresetConfig | None = None,
) -> MultiChamberParliamentModel:
    """Polish Parliament (Sejm + Senat).

    - **Sejm** (460 seats) – simple majority passage; can override Senat
      rejection with an *absolute majority* (231 votes out of 460).
    - **Senat** (100 seats) – suspensive veto; Sejm can override.
    - The Senat has 30 days to amend or reject; if it does reject, the Sejm
      votes on whether to override (modelled as override_threshold = 0.5).

    Constitutional reference:
        Articles 119–127, Konstytucja Rzeczypospolitej Polskiej (1997).
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "Parlament RP"
    policy_dim = cfg.policy_dim
    sejm_size = cfg.lower_house_size or 460
    senat_size = cfg.upper_house_size or 100

    sejm = _make_chamber(sejm_size, policy_dim, "Poseł")
    senat = _make_chamber(senat_size, policy_dim, "Senator")

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        sejm,
        ChamberConfig(
            name="Sejm",
            role=ChamberRole.LOWER,
            size=sejm_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    parliament.add_chamber(
        senat,
        ChamberConfig(
            name="Senat",
            role=ChamberRole.UPPER,
            size=senat_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
            powers=UpperChamberPowers.OVERRIDE_BY_LOWER,
            # Absolute majority of Sejm = 231/460 ≈ 0.502 of total seats
            override_threshold=231 / 460,
        ),
    )
    return parliament


def create_swedish_parliament(
    config: ParliamentPresetConfig | None = None,
) -> MultiChamberParliamentModel:
    """Swedish Parliament – Riksdag (unicameral since 1971).

    - **Riksdag** (349 seats) – simple majority; qualified majority (3/5) for
      constitutional amendments.

    Constitutional reference:
        Instrument of Government (Regeringsformen) 1974, Sweden.
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "Riksdag"
    policy_dim = cfg.policy_dim
    riksdag_size = cfg.lower_house_size or 349

    riksdag = _make_chamber(riksdag_size, policy_dim, "Riksdagsledamot")

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        riksdag,
        ChamberConfig(
            name="Riksdag",
            role=ChamberRole.UNICAMERAL,
            size=riksdag_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    return parliament


def create_spanish_parliament(
    config: ParliamentPresetConfig | None = None,
) -> MultiChamberParliamentModel:
    """Spanish Parliament (Cortes Generales) – Congreso + Senado.

    - **Congreso de los Diputados** (350 seats) – primary legislative chamber;
      can override Senate veto with absolute majority, and with simple majority
      after two months.
    - **Senado** (265 seats) – suspensive veto; can amend or veto bills
      but Congreso always has the final word.

    Constitutional reference:
        Articles 74–91, Constitución Española (1978).
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "Cortes Generales"
    policy_dim = cfg.policy_dim
    congreso_size = cfg.lower_house_size or 350
    senado_size = cfg.upper_house_size or 265

    congreso = _make_chamber(congreso_size, policy_dim, "Diputado")
    senado = _make_chamber(senado_size, policy_dim, "Senador")

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        congreso,
        ChamberConfig(
            name="Congreso de los Diputados",
            role=ChamberRole.LOWER,
            size=congreso_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    parliament.add_chamber(
        senado,
        ChamberConfig(
            name="Senado",
            role=ChamberRole.UPPER,
            size=senado_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
            powers=UpperChamberPowers.SUSPENSIVE_VETO,
            max_ping_pong_rounds=1,  # Congreso overrides after one rejection
        ),
    )
    return parliament


def create_australian_parliament(
    config: ParliamentPresetConfig | None = None,
) -> MultiChamberParliamentModel:
    """Australian Parliament – House of Representatives + Senate.

    - **House of Representatives** (151 seats) – simple majority.
    - **Senate** (76 seats) – symmetric chamber; full veto (both must agree).
    - Deadlock resolution: double dissolution election (not modelled here).

    Constitutional reference:
        Sections 53–60, Commonwealth of Australia Constitution Act (1900).
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "Parliament of Australia"
    policy_dim = cfg.policy_dim
    house_size = cfg.lower_house_size or 151
    senate_size = cfg.upper_house_size or 76

    house = _make_chamber(house_size, policy_dim, "MHR")
    senate = _make_chamber(senate_size, policy_dim, "Senator")

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        house,
        ChamberConfig(
            name="House of Representatives",
            role=ChamberRole.LOWER,
            size=house_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    parliament.add_chamber(
        senate,
        ChamberConfig(
            name="Senate",
            role=ChamberRole.UPPER,
            size=senate_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
            powers=UpperChamberPowers.FULL_VETO,
        ),
    )
    return parliament


def create_canadian_parliament(
    config: ParliamentPresetConfig | None = None,
) -> MultiChamberParliamentModel:
    """Canadian Parliament – House of Commons + Senate (appointed upper house).

    - **House of Commons** (338 seats) – simple majority; primary chamber.
    - **Senate of Canada** (105 seats, appointed) – formally full veto but in
      practice acts as a revising/advisory chamber. Modelled here as
      ``SUSPENSIVE_VETO`` with a single round (reflects political convention).

    Constitutional reference:
        Constitution Act, 1867 (formerly British North America Act).
    """
    cfg = config or ParliamentPresetConfig()
    name = cfg.parliament_name or "Parliament of Canada"
    policy_dim = cfg.policy_dim
    commons_size = cfg.lower_house_size or 338
    senate_size = cfg.upper_house_size or 105

    commons = _make_chamber(commons_size, policy_dim, "MP")
    senate = _make_chamber(senate_size, policy_dim, "Senator")

    parliament = MultiChamberParliamentModel(name)
    parliament.add_chamber(
        commons,
        ChamberConfig(
            name="House of Commons",
            role=ChamberRole.LOWER,
            size=commons_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
        ),
    )
    parliament.add_chamber(
        senate,
        ChamberConfig(
            name="Senate of Canada",
            role=ChamberRole.UPPER,
            size=senate_size,
            passage_threshold=PassageThreshold.SIMPLE_MAJORITY,
            powers=UpperChamberPowers.SUSPENSIVE_VETO,
            max_ping_pong_rounds=1,
        ),
    )
    return parliament


# ---------------------------------------------------------------------------
# Convenience mapping
# ---------------------------------------------------------------------------

#: All available parliament presets keyed by short identifier.
PARLIAMENT_PRESETS: dict[str, Any] = {
    "uk": create_uk_parliament,
    "us": create_us_congress,
    "germany": create_german_parliament,
    "france": create_french_parliament,
    "italy": create_italian_parliament,
    "poland": create_polish_parliament,
    "sweden": create_swedish_parliament,
    "spain": create_spanish_parliament,
    "australia": create_australian_parliament,
    "canada": create_canadian_parliament,
}


def list_presets() -> list[str]:
    """Return a sorted list of available preset identifiers."""
    return sorted(PARLIAMENT_PRESETS.keys())


def create_parliament(
    preset: str,
    config: ParliamentPresetConfig | None = None,
    **kwargs: Any,
) -> MultiChamberParliamentModel:
    """Create a parliament by preset identifier.

    Parameters
    ----------
    preset:
        One of the keys returned by :func:`list_presets`.
    config:
        Optional :class:`ParliamentPresetConfig` for customisation.
    **kwargs:
        Extra keyword arguments forwarded to the specific factory
        (e.g. ``consent_law=False`` for Germany, ``max_navette_rounds=3`` for France).

    Raises
    ------
    ValueError
        If *preset* is not recognised.
    """
    if preset not in PARLIAMENT_PRESETS:
        available = ", ".join(sorted(PARLIAMENT_PRESETS.keys()))
        raise ValueError(f"Unknown parliament preset {preset!r}. Available: {available}")
    factory = PARLIAMENT_PRESETS[preset]
    if kwargs:
        return factory(config, **kwargs)
    return factory(config)
