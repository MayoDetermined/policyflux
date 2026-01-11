"""Entry point for the congressional simulation.

Now delegates to the BehavioralSim compiler/runner API for a TensorFlow-like
compile → run workflow while keeping the previous CLI surface compatible.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import logging
from typing import Any, Dict, List, Optional, Tuple

from policyflux import config
from policyflux.behavioral_sim.api import CongressCompiler, CongressRunner
from policyflux.behavioral_sim.network import HomophilyInfluence, LeaderBoostInfluence, CommitteeInfluence


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure basic logging based on config settings."""
    logging.basicConfig(
        level=getattr(logging, config.LOGGING_LEVEL.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _print_instability_report(report: Dict[str, Any]) -> None:
    """Emit a concise summary of key instability metrics."""
    if not report:
        return
    logger.info("Vote Volatility (Std): %.4f", report.get("vote_volatility", 0.0))
    logger.info("Risk of Razor-Thin Outcome: %.2f%%", report.get("risk_of_flip", 0.0) * 100)
    logger.info("-" * 60)


def _build_compiler(use_gpu: bool, use_hmm_state: bool, scenario: Optional[str]) -> CongressCompiler:
    compiler = CongressCompiler(device="cuda" if use_gpu else "cpu")
    compiler.add_influence_function(HomophilyInfluence(beta=config.HOMOPHILY_BETA))
    compiler.add_influence_function(LeaderBoostInfluence(boost=config.LEADER_BOOST))
    compiler.add_influence_function(CommitteeInfluence(weight=config.COMMITTEE_WEIGHT))
    return compiler


def run_full_simulation(
    use_cache: bool = True,
    scenario: Optional[str] = None,
    n_simulations: Optional[int] = None,
    steps: Optional[int] = None,
    use_gpu: bool = True,
    use_hmm_state: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    show_tqdm: bool = True,
    show_small_steps: bool = True,
    show_step_tqdm: bool = True,
    show_component_tqdm: bool = True,
) -> Tuple[Dict[str, Any], Optional[object], List[Any]]:

    """Build, fit, and simulate a Congress instance via BehavioralSim.

    The signature is backward-compatible with the previous CongressSimulator-based
    implementation but now routes through CongressCompiler/CongressRunner.
    """

    compiler = _build_compiler(use_gpu=use_gpu, use_hmm_state=use_hmm_state, scenario=scenario)
    compiled = compiler.compile(
        use_cache=use_cache,
        scenario=scenario or config.SCENARIO,
        use_gpu=use_gpu,
        use_hmm_state=use_hmm_state,
    )

    runner = CongressRunner(compiled)
    report = runner.run_monte_carlo(
        n_simulations=n_simulations or config.NUM_SIMULATIONS,
        steps=steps or config.SIMULATION_STEPS,
        progress_callback=progress_callback,
        show_tqdm=show_tqdm,
        show_small_steps=show_small_steps,
        show_step_tqdm=show_step_tqdm,
        show_component_tqdm=show_component_tqdm,
    )

    _print_instability_report(report)
    return report, compiled.congress, compiled.actors


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entrypoint; returns the simulation report for programmability."""

    _configure_logging()

    parser = argparse.ArgumentParser(description="Run congressional dynamics simulation")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable cached models and reload data")
    parser.add_argument("--scenario", type=str, default=None, help="Override scenario (stable|polarized|crisis)")
    parser.add_argument("--simulations", type=int, default=None, help="Number of Monte Carlo runs")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps per simulation")
    parser.add_argument("--cpu", dest="use_gpu", action="store_false", help="Force CPU even if GPU is available")
    parser.add_argument("--no-hmm", dest="use_hmm_state", action="store_false", help="Disable HMM-based regime updates")
    parser.add_argument("--progress", dest="progress", action="store_true", help="Print progress events during simulation")
    parser.add_argument("--show-tqdm", dest="show_tqdm", action="store_true", help="Show tqdm progress bars if available")
    parser.add_argument("--show-small-steps", dest="show_small_steps", action="store_true", help="Emit per-step timing events for every step (useful for debugging)")
    parser.add_argument("--show-step-tqdm", dest="show_step_tqdm", action="store_true", help="Show an inner tqdm over steps for each simulation")
    parser.add_argument("--show-component-tqdm", dest="show_component_tqdm", action="store_true", help="Show per-component tqdm (actor-level) for evolution and voting")

    # Default to maximum verbosity and progress reporting
    parser.set_defaults(use_cache=True, use_gpu=True, use_hmm_state=True, progress=True, show_tqdm=True, show_small_steps=True, show_step_tqdm=True, show_component_tqdm=True)

    args = parser.parse_args(argv)

    def _print_progress(ev: dict) -> None:
        try:
            logger.info("PROGRESS: %s", ev)
        except Exception:
            pass

    progress_cb = _print_progress if args.progress else None

    report, _, _ = run_full_simulation(
        use_cache=args.use_cache,
        scenario=args.scenario,
        n_simulations=args.simulations,
        steps=args.steps,
        use_gpu=args.use_gpu,
        use_hmm_state=args.use_hmm_state,
        progress_callback=progress_cb,
        show_tqdm=args.show_tqdm,
        show_small_steps=args.show_small_steps,
        show_step_tqdm=args.show_step_tqdm,
        show_component_tqdm=args.show_component_tqdm,
    )

    _print_instability_report(report)
    return report


if __name__ == "__main__":
    main(sys.argv[1:])



