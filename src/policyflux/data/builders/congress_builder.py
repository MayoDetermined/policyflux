"""Congressional System Builder - Separated from Simulation Logic.

This module handles:
- Fetching voting data from VoteView
- Training ML models (autoencoders, ideal point models)
- Computing behavioral features (loyalty, volatility)
- Building Congress actor networks

It is separate from simulation logic to maintain clean separation of concerns.

For backward compatibility, this re-exports CongressMenBuilder from the original module.
"""
from policyflux.data.collectors.actors_architectural_bureau import CongressMenBuilder

# Alias for cleaner API
CongressBuilder = CongressMenBuilder

__all__ = ["CongressBuilder", "CongressMenBuilder"]
