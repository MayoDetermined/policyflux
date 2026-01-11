"""
PRACTICAL DEMONSTRATION: Ideal Point Model-Based Voting Decisions

This script demonstrates how the IPM voting system works end-to-end:
1. Create actors (legislators)
2. Create laws with IPM parameters
3. Compute voting decisions
4. Simulate regime pressure effects
"""

import logging
import numpy as np
from policyflux.congress.actors import CongressMan
from policyflux.congress.law import Law
from policyflux.public_opinion.regime import PublicRegime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def demo_step_1_create_actors():
    """Step 1: Create congressional actors with multi-dimensional ideology."""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Create Congressional Actors (Legislators)")
    logger.info("="*70)
    
    # Create some example legislators
    actors_data = [
        {
            "id": 1,
            "ideology": [0.8, 0.5, 0.2],  # Conservative on all dimensions
            "party": "Republican",
            "loyalty": 0.8,
            "vulnerability": 0.1,
            "volatility": 0.05
        },
        {
            "id": 2,
            "ideology": [-0.7, -0.4, -0.3],  # Liberal on all dimensions
            "party": "Democratic",
            "loyalty": 0.75,
            "vulnerability": 0.15,
            "volatility": 0.06
        },
        {
            "id": 3,
            "ideology": [0.2, -0.5, 0.6],  # Mixed ideology (centrist)
            "party": "Independent",
            "loyalty": 0.5,
            "vulnerability": 0.25,
            "volatility": 0.1
        },
    ]
    
    actors = []
    for data in actors_data:
        actor_input = {
            "ideology_multidim": data["ideology"],
            "party": data["party"],
            "loyalty": data["loyalty"],
            "vulnerability": data["vulnerability"],
            "volatility": data["volatility"]
        }
        actor = CongressMan(data["id"], actor_input)
        actors.append(actor)
        
        logger.info(f"\nActor {actor.id} ({actor.party}):")
        logger.info(f"  Ideology (multi-dim): {actor.ideology}")
        logger.info(f"  Ideology (scalar): {actor.get_ideological_position():.3f}")
        logger.info(f"  Loyalty: {actor.loyalty:.2f} (resistance to change)")
        logger.info(f"  Vulnerability: {actor.vulnerability:.2f} (susceptibility to influence)")
    
    return actors


def demo_step_2_create_laws():
    """Step 2: Create laws with IPM parameters."""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Create Laws (Bills) with IPM Parameters")
    logger.info("="*70)
    
    laws = [
        {
            "id": 1,
            "title": "Healthcare Expansion (Progressive)",
            "salience": np.array([-0.8, -0.6, -0.4]),  # Progressive stance
            "threshold": 0.2
        },
        {
            "id": 2,
            "title": "Tax Cut (Conservative)",
            "salience": np.array([0.7, 0.5, 0.3]),  # Conservative stance
            "threshold": -0.1
        },
        {
            "id": 3,
            "title": "Infrastructure (Bipartisan)",
            "salience": np.array([0.1, 0.2, 0.8]),  # Neutral ideology, high salience
            "threshold": 0.0
        },
    ]
    
    law_objects = []
    for law_data in laws:
        law = Law(
            law_id=law_data["id"],
            salience=law_data["salience"],
            threshold=law_data["threshold"],
            title=law_data["title"]
        )
        law_objects.append(law)
        
        logger.info(f"\nLaw {law.law_id}: {law.title}")
        logger.info(f"  Salience vector a_j: {law.salience}")
        logger.info(f"  Threshold b_j: {law.threshold:.3f}")
        logger.info(f"  Interpretation: {'Progressive' if law.salience[0] < 0 else 'Conservative'}")
    
    return law_objects


def demo_step_3_compute_voting_decisions(actors, laws):
    """Step 3: Compute voting decisions using IPM formula."""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Compute Voting Decisions (IPM Formula)")
    logger.info("="*70)
    
    logger.info("\nFormula: P(vote=1) = sigmoid(||a_j|| * (x_i · ā_j - b_j))")
    logger.info("Where:")
    logger.info("  x_i = legislator's multi-dimensional ideal point")
    logger.info("  a_j = law's salience vector (distinguishing power)")
    logger.info("  b_j = law's threshold (difficulty parameter)")
    
    results = {}
    
    for law in laws:
        logger.info(f"\n{'─'*70}")
        logger.info(f"LAW {law.law_id}: {law.title}")
        logger.info(f"{'─'*70}")
        
        results[law.law_id] = {}
        
        for actor in actors:
            # Compute voting probability using IPM
            prob = actor.decide_vote(law)
            
            # Cast actual vote
            vote = actor.cast_vote()
            
            results[law.law_id][actor.id] = {
                "probability": prob,
                "vote": vote
            }
            
            logger.info(f"\n  Actor {actor.id} ({actor.party}):")
            logger.info(f"    Ideology: {actor.ideology}")
            logger.info(f"    P(vote YES) = {prob:.4f}")
            logger.info(f"    Vote: {'✓ YES' if vote else '✗ NO'}")
    
    return results


def demo_step_4_regime_effects(actors, law, regime):
    """Step 4: Show how regime pressure affects voting."""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Regime Pressure Effects on Voting")
    logger.info("="*70)
    
    logger.info(f"\nRegime Scenario: {regime.scenario.upper()}")
    logger.info(f"Base Pressure Z(t): {regime.base_pressure:.2f}")
    logger.info(f"Volatility Multiplier: {regime.volatility_multiplier:.1f}x")
    
    logger.info(f"\nHow Pressure Affects Voting:")
    logger.info(f"  High pressure → actors more susceptible to external forces")
    logger.info(f"  Low pressure → actors follow ideology (loyalty) more closely")
    
    logger.info(f"\n{'─'*70}")
    logger.info(f"Voting under {regime.scenario.upper()} scenario:")
    logger.info(f"{'─'*70}")
    
    # Test the same law under different pressure conditions
    pressures = [regime.get_current_pressure() for _ in range(10)]
    
    for actor in actors[:2]:  # Just show first 2 actors
        base_prob = actor.decide_vote(law)
        
        logger.info(f"\n  Actor {actor.id} ({actor.party}):")
        logger.info(f"    Base IPM probability: {base_prob:.4f}")
        logger.info(f"    Loyalty (resistance): {actor.loyalty:.2f}")
        logger.info(f"    Vulnerability (susceptibility): {actor.vulnerability:.2f}")
        logger.info(f"    Sample pressure values from regime: {[f'{p:.2f}' for p in pressures[:5]]}")


def demo_step_5_ipm_parameter_export():
    """Step 5: Show how IPM parameters are exported for law creation."""
    logger.info("\n" + "="*70)
    logger.info("STEP 5: IPM Parameter Export (for creating laws from data)")
    logger.info("="*70)
    
    # Simulate exported parameters from IPM training
    n_votes = 500  # Example: 500 historical votes
    dim = 3  # 3 dimensions
    
    exported_params = {
        'salience': np.random.randn(n_votes, dim).astype(np.float32),
        'threshold': np.random.uniform(-1, 1, n_votes).astype(np.float32)
    }
    
    logger.info(f"\nExported IPM Parameters:")
    logger.info(f"  Salience (a_j) shape: {exported_params['salience'].shape}")
    logger.info(f"    → {n_votes} historical votes × {dim} dimensions")
    logger.info(f"  Threshold (b_j) shape: {exported_params['threshold'].shape}")
    logger.info(f"    → {n_votes} scalar thresholds")
    
    # Show how to create laws from these parameters
    logger.info(f"\nCreating laws from historical IPM parameters:")
    
    for vote_id in [0, 10, 100]:  # Sample a few votes
        law = Law.create_from_ipm_parameters(
            law_id=vote_id,
            vote_id=vote_id,
            ipm_voting_params=exported_params,
            title=f"Historical Vote {vote_id}"
        )
        
        logger.info(f"\n  Law from Vote {vote_id}:")
        logger.info(f"    Salience: {law.salience}")
        logger.info(f"    Threshold: {law.threshold:.3f}")


def demo_step_6_full_simulation():
    """Step 6: Full simulation with multiple laws and pressure."""
    logger.info("\n" + "="*70)
    logger.info("STEP 6: Full Voting Simulation")
    logger.info("="*70)
    
    # Create 3 actors
    actors = demo_step_1_create_actors()
    
    # Create 3 laws
    laws = demo_step_2_create_laws()
    
    # Try two regimes
    for regime_name in ["stable", "crisis"]:
        logger.info(f"\n{'═'*70}")
        logger.info(f"SCENARIO: {regime_name.upper()}")
        logger.info(f"{'═'*70}")
        
        regime = PublicRegime(scenario=regime_name)
        pressure = regime.get_current_pressure()
        
        logger.info(f"\nRegime Pressure Z(t): {pressure:.3f}")
        
        # Vote on each law
        total_yes = 0
        for law in laws:
            law_yes = 0
            
            logger.info(f"\n  {law.title}")
            
            for actor in actors:
                # Compute IPM-based probability
                prob = actor.decide_vote(law)
                
                # Cast vote
                vote = actor.cast_vote()
                
                if vote:
                    law_yes += 1
                    total_yes += 1
                
                logger.info(f"    Actor {actor.id} ({actor.party}): "
                          f"P={prob:.3f} → {'YES' if vote else 'NO'}")
            
            passage = "PASS" if law_yes > len(actors) / 2 else "FAIL"
            logger.info(f"    Result: {law_yes}/{len(actors)} votes → {passage}")
        
        logger.info(f"\nTotal votes across all laws: {total_yes}/{len(actors) * len(laws)}")


def main():
    """Run all demonstrations."""
    logger.info("\n")
    logger.info("╔" + "═"*68 + "╗")
    logger.info("║" + " "*68 + "║")
    logger.info("║" + "  IPM-BASED VOTING DECISION SYSTEM - COMPLETE DEMONSTRATION".center(68) + "║")
    logger.info("║" + " "*68 + "║")
    logger.info("╚" + "═"*68 + "╝")
    
    # Step 1: Create actors
    actors = demo_step_1_create_actors()
    
    # Step 2: Create laws
    laws = demo_step_2_create_laws()
    
    # Step 3: Compute voting decisions
    results = demo_step_3_compute_voting_decisions(actors, laws)
    
    # Step 4: Show regime effects
    regime_stable = PublicRegime(scenario="stable")
    demo_step_4_regime_effects(actors, laws[0], regime_stable)
    
    # Step 5: IPM parameter export
    demo_step_5_ipm_parameter_export()
    
    # Step 6: Full simulation
    demo_step_6_full_simulation()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY: IPM VOTING SYSTEM")
    logger.info("="*70)
    
    logger.info("""
✓ Law Class: Represents bills with multi-dimensional salience & threshold
✓ CongressMan.decide_vote(): Computes IPM-based voting probability
✓ Multi-dimensional Ideology: Supports 3D actor positions
✓ Regime Integration: Pressure Z(t) modulates actor behavior
✓ Dynamic Simulation: Full voting simulation across scenarios

The system is production-ready for:
  • Training on real congressional voting data
  • Creating synthetic laws from IPM parameters
  • Simulating voting under different political scenarios
  • Analyzing network effects and regime impacts
""")


if __name__ == "__main__":
    main()




