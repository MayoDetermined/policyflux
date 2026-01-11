"""BehavioralSim public API shim.

This package wraps the existing congressional simulator into a modular,
TensorFlow-like compile/run workflow with pluggable agents, influence
functions, and evolution models.
"""

from behavioral_sim.api import CongressCompiler, CongressRunner, CompiledSystem
from behavioral_sim.agents.base import BehavioralParameters, AbstractAgent
from behavioral_sim.network.influence import (
    InfluenceFunction,
    HomophilyInfluence,
    LeaderBoostInfluence,
    CommitteeInfluence,
)

__all__ = [
    "CongressCompiler",
    "CongressRunner",
    "CompiledSystem",
    "BehavioralParameters",
    "AbstractAgent",
    "InfluenceFunction",
    "HomophilyInfluence",
    "LeaderBoostInfluence",
    "CommitteeInfluence",
]
