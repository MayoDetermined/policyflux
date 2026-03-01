from ...core.container import ServiceContainer
from ...core.id_generator import IdGenerator, get_id_generator
from ...engines.sequential_monte_carlo import SequentialMonteCarlo
from ...engines.session_management import Session
from ...pfrandom import set_seed
from ...toolbox.bill_models import SequentialBill
from ..config import IntegrationConfig
from .congress_builder import build_congress


def _create_container(config: IntegrationConfig) -> ServiceContainer:
    """Create and populate a service container for dependency management."""
    container = ServiceContainer()
    container.register_singleton(IntegrationConfig, config)
    container.register_singleton(IdGenerator, get_id_generator())
    return container


def build_bill(config: IntegrationConfig) -> SequentialBill:
    bill = SequentialBill(id=None)
    bill.make_random_position(dim=config.policy_dim)
    return bill


def build_session(config: IntegrationConfig) -> Session:
    set_seed(config.seed)
    congress = build_congress(config)
    bill = build_bill(config)
    return Session(
        n=config.iterations,
        seed=config.seed,
        bill=bill,
        description=config.description,
        congress_model=congress,
    )


def build_engine(config: IntegrationConfig) -> SequentialMonteCarlo:
    """Build a complete simulation engine from configuration.

    Creates a ServiceContainer internally to manage shared instances
    (IdGenerator, config) throughout the build pipeline.
    """
    container = _create_container(config)
    set_seed(config.seed)
    session = build_session(container.resolve(IntegrationConfig))
    return SequentialMonteCarlo(session_params=session)
