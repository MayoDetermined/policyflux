"""Re-export legacy model classes for convenience."""

try:
    from models.ideal_point import IdealPointModel  # type: ignore
except Exception:  # pragma: no cover
    IdealPointModel = None  # type: ignore

try:
    from models.dbn import DBCongressModel  # type: ignore
except Exception:  # pragma: no cover
    DBCongressModel = None  # type: ignore

try:
    from models.dqn import VoteDQN, DQNAgent  # type: ignore
except Exception:  # pragma: no cover
    VoteDQN = None  # type: ignore
    DQNAgent = None  # type: ignore

try:
    from models.rnn import ActorLSTM, ActorLSTMTrainer  # type: ignore
except Exception:  # pragma: no cover
    ActorLSTM = None  # type: ignore
    ActorLSTMTrainer = None  # type: ignore

__all__ = [
    "IdealPointModel",
    "DBCongressModel",
    "VoteDQN",
    "DQNAgent",
    "ActorLSTM",
    "ActorLSTMTrainer",
]
