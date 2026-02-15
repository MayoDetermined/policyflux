from typing import Optional, List
from ..config import IntegrationConfig
from ...core.executive import ExecutiveType, Executive
from ...toolbox.executive_systems import (
    PresidentialExecutive,
    ParliamentaryExecutive,
    SemiPresidentialExecutive,
    President,
    PrimeMinister,
)
from ...toolbox.advanced_actors.lobby import SequentialLobbyer
from ...toolbox.advanced_actors.whips import SequentialWhip
from ...toolbox.advanced_actors.speaker import SequentialSpeaker
from ...toolbox.advanced_actors.white_house import SequentialPresident

def build_executive(config: IntegrationConfig) -> Optional[Executive]:
    """Build executive branch based on configuration.

    Args:
        config: Integration configuration with actors_config

    Returns:
        Executive instance (Presidential, Parliamentary, or SemiPresidential) or None
    """
    exec_type = config.actors_config.executive_type

    if exec_type == ExecutiveType.PRESIDENTIAL:
        president = President(
            approval_rating=config.actors_config.president_approval_rating,
            name="President"
        )
        return PresidentialExecutive(
            president=president,
            veto_override_threshold=config.actors_config.veto_override_threshold
        )

    elif exec_type == ExecutiveType.PARLIAMENTARY:
        prime_minister = PrimeMinister(
            party_strength=config.actors_config.pm_party_strength,
            name="PrimeMinister"
        )
        # Auto-enable government agenda layer for parliamentary systems
        if not config.layer_config.include_government_agenda:
            config.layer_config.include_government_agenda = True
            config.layer_config.government_agenda_pm_strength = config.actors_config.pm_party_strength

        return ParliamentaryExecutive(
            prime_minister=prime_minister,
            confidence_threshold=config.actors_config.confidence_threshold
        )

    elif exec_type == ExecutiveType.SEMI_PRESIDENTIAL:
        president = President(
            approval_rating=config.actors_config.semi_president_approval,
            name="President"
        )
        prime_minister = PrimeMinister(
            party_strength=config.actors_config.semi_pm_party_strength,
            name="PrimeMinister"
        )
        return SemiPresidentialExecutive(
            president=president,
            prime_minister=prime_minister
        )

    return None


def build_advanced_actors(config: IntegrationConfig) -> tuple[List[SequentialLobbyer], List[SequentialWhip], SequentialSpeaker, SequentialPresident]:
    lobbyists = [
        SequentialLobbyer(
            influence_strength=config.actors_config.lobbyist_strength,
            stance=config.actors_config.lobbyist_stance,
            name=f"Lobbyer_{i + 1}",
        )
        for i in range(config.actors_config.n_lobbyists)
    ]

    whips = [
        SequentialWhip(
            discipline_strength=config.actors_config.whip_discipline_strength,
            party_line_support=config.actors_config.whip_party_line_support,
            name=f"Whip_{i + 1}",
        )
        for i in range(config.actors_config.n_whips)
    ]

    speaker = SequentialSpeaker(agenda_support=config.actors_config.speaker_agenda_support)
    president = SequentialPresident(approval_rating=config.actors_config.president_approval_rating)

    return lobbyists, whips, speaker, president