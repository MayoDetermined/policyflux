"""PolicyFlux public API.

This module re-exports the primary simulation entrypoints so user code can
quickly build and run congressional dynamics experiments.
"""
from __future__ import annotations

from behavioral_sim.api import CongressCompiler, CongressRunner
from behavioral_sim.network import CommitteeInfluence, HomophilyInfluence, LeaderBoostInfluence
from congress_simulator import CongressSimulator
from main import run_full_simulation
from policyflux.core import (
    Accuracy,
    Adam,
    CrossEntropy,
    EarlyStopping,
    Layer,
    Model,
    ModelCheckpoint,
    MSE,
    Precision,
    Recall,
    Sequential,
    SGD,
)
from policyflux.layers import (
    ActorLayer,
    NetworkInfluenceLayer,
    RegimeContextLayer,
    VotingLayer,
)

__all__ = [
    # Legacy API
    "CongressSimulator",
    "CongressCompiler",
    "CongressRunner",
    "HomophilyInfluence",
    "LeaderBoostInfluence",
    "CommitteeInfluence",
    "run_full_simulation",
    # TensorFlow-like API - Core
    "Model",
    "Sequential",
    "Layer",
    "Adam",
    "SGD",
    "MSE",
    "CrossEntropy",
    "Accuracy",
    "Precision",
    "Recall",
    "EarlyStopping",
    "ModelCheckpoint",
    # TensorFlow-like API - Behavioral Layers
    "ActorLayer",
    "NetworkInfluenceLayer",
    "VotingLayer",
    "RegimeContextLayer",
]
