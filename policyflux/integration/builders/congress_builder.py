from ...toolbox.actors import SequentialVoter
from ...toolbox.congress_model import SequentialCongressModel
from ..config import IntegrationConfig
from .actor_builder import build_advanced_actors, build_executive
from .layer_builder import build_layers
from .mechanics_builders import build_aggregation_strategy


def build_congress(config: IntegrationConfig) -> SequentialCongressModel:
    lobbyists, whips, speaker, president = build_advanced_actors(config)
    aggregation_strategy = build_aggregation_strategy(config)
    executive = build_executive(config)

    congress = SequentialCongressModel(id=None)
    for i in range(1, config.num_actors + 1):
        voter_layers = build_layers(config, lobbyists, whips)
        voter = SequentialVoter(
            id=None,
            name=f"Rep-{i}",
            layers=voter_layers,
            aggregation_strategy=aggregation_strategy,
        )
        congress.add_congressman(voter)

    congress.lobbyists = lobbyists
    for whip in whips:
        congress.add_whip(whip)

    congress.set_speaker(speaker)
    congress.set_president(president)

    # Set executive system if built
    if executive is not None:
        congress.set_executive(executive)

    congress.compile()
    return congress
