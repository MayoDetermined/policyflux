"""BehavioralSim public API shim.

This package wraps the existing congressional simulator into a modular,
TensorFlow-like compile/run workflow with pluggable agents, influence
functions, and evolution models.
"""

from policyflux.behavioral_sim.api import CongressCompiler, CongressRunner, CompiledSystem
from policyflux.behavioral_sim.agents.base import BehavioralParameters, AbstractAgent
from policyflux.behavioral_sim.network.influence import (
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




