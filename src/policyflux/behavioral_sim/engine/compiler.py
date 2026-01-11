from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import tensorflow as tf

from policyflux.behavioral_sim.agents.congressman import CongressAgent
from policyflux.behavioral_sim.context import PublicRegime
from policyflux.behavioral_sim.network import DynamicNetwork, InfluenceFunction
from policyflux.congress_simulator import CongressSimulator
from policyflux.gpu_utils import get_tf_device


@dataclass
class CompiledSystem:
    """Container for compiled components ready to run."""

    actors: List[CongressAgent]
    regime: PublicRegime
    congress: Any
    dynamic_network: Optional[DynamicNetwork]
    engine: Any
    device: str
    ml_models: Dict[str, Any] = field(default_factory=dict)
    evolution_model: Optional[Any] = None


class CongressCompiler:
    """Configures models, influence functions, and evolution, then compiles into a runnable system."""

    def __init__(self, device: Optional[str] = None) -> None:
        # Resolve preferred device string and fall back to CPU when GPU unavailable.
        if device:
            dev_str = device.strip().lower()
            if dev_str in {"cuda", "gpu"}:
                self.device = get_tf_device(prefer_gpu=True)
            elif dev_str.startswith("/gpu") or dev_str.startswith("gpu"):
                self.device = "/GPU:0"
            elif dev_str == "cpu" or dev_str.startswith("/cpu"):
                self.device = "/CPU:0"
            else:
                self.device = get_tf_device()
        else:
            self.device = get_tf_device()
        self.ml_models: Dict[str, Dict[str, Any]] = {}
        self.influence_functions: List[InfluenceFunction] = []
        self.evolution_model_cls: Optional[Callable[..., Any]] = None
        self.evolution_model_config: Dict[str, Any] = {}

    def add_ml_model(self, name: str, model_cls: Type, config: Optional[Dict[str, Any]] = None) -> "CongressCompiler":
        self.ml_models[name] = {"cls": model_cls, "config": config or {}}
        return self

    def add_influence_function(self, influence_fn: InfluenceFunction) -> "CongressCompiler":
        self.influence_functions.append(influence_fn)
        return self

    def set_evolution_model(self, model_cls: Callable[..., Any], config: Optional[Dict[str, Any]] = None) -> "CongressCompiler":
        self.evolution_model_cls = model_cls
        self.evolution_model_config = config or {}
        return self

    def compile(
        self,
        data_source: Optional[Any] = None,
        use_cache: bool = True,
        scenario: Optional[str] = None,
        use_gpu: bool = True,
        use_hmm_state: bool = True,
    ) -> CompiledSystem:
        """Build actors, congress, engine, and dynamic network.

        Falls back to the legacy CongressSimulator for data preparation and
        engine construction, then wraps the result with influence functions.
        """

        simulator = CongressSimulator(
            scenario=scenario,
            use_gpu=use_gpu,
            use_hmm_state=use_hmm_state,
        )
        simulator.compile()
        simulator.fit(use_cache=use_cache)

        # Upgrade actors to CongressAgent to expose params
        actors: List[CongressAgent] = []
        for a in simulator.actors:
            agent = CongressAgent(a.id, data=getattr(a, "__dict__", {}), ideal_point_model=a.ideal_point_model)
            agents_params = getattr(a, "params", None)
            if agents_params is not None:
                agent.params = agents_params
            actors.append(agent)
        simulator.actors = actors
        if simulator.congress is not None:
            simulator.congress.actors = actors
            simulator.engine.template = simulator.congress

        dynamic_network: Optional[DynamicNetwork] = None
        if simulator.congress is not None:
            base_adj = tf.convert_to_tensor(simulator.congress.base_adj_matrix, dtype=tf.float32)
            dynamic_network = DynamicNetwork(
                base_adj=base_adj,
                influence_functions=list(self.influence_functions),
                device=self.device,
            )

        evolution_model = None
        if self.evolution_model_cls is not None:
            try:
                evolution_model = self.evolution_model_cls(**self.evolution_model_config)
            except Exception:
                evolution_model = None

        compiled = CompiledSystem(
            actors=actors,
            regime=simulator.regime,
            congress=simulator.congress,
            dynamic_network=dynamic_network,
            engine=simulator.engine,
            device=self.device,
            ml_models={name: cfg for name, cfg in self.ml_models.items()},
            evolution_model=evolution_model,
        )
        return compiled




