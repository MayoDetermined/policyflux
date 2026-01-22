#!/usr/bin/env python3
"""
Simple Congress Simulation
Demonstrates basic usage of the Congress refactor library
"""

import random
from policyflux.models import (
    SequentialCongressModel,
    SequentialVoter,
    SequentialBill,
    Session,
    SequentialMonteCarlo,
)
from policyflux.layers import IdealPointEncoder, LobbyingLayer, PublicOpinionLayer

# Configuration
SEED = 42
NUM_ACTORS = 461
POLICY_SPACE_DIM = 3

# Set random seed for reproducibility
random.seed(SEED)

print("=" * 60)
print("Congressional Voting Simulation")
print("=" * 60)

# Step 1: Create Congress
print(f"\n[1] Creating Congress with {NUM_ACTORS} members...")
congress = SequentialCongressModel(id=1)

# Step 2: Add representatives with layers
print(f"[2] Adding representatives with behavior layers...")
for i in range(1, NUM_ACTORS + 1):
    actor = SequentialVoter(
        id=i,
        name=f"Rep. #{i}",
        layers=[
            IdealPointEncoder(
                id=1000 + i,
                space=[random.random() for _ in range(POLICY_SPACE_DIM)],
                status_quo=[0.5] * POLICY_SPACE_DIM
            ),
            LobbyingLayer(id=2000 + i, intensity=random.uniform(0.0, 0.3)),
            PublicOpinionLayer(id=3000 + i, support_level=random.random())
        ]
    )
    congress.add_congressman(actor)
    if (i) % 50 == 0:
        print(f"   Added {i} representatives...")

congress.compile()
print(f"   Congress compiled successfully with {NUM_ACTORS} members")

# Step 3: Create a bill
print(f"\n[3] Creating bill for voting...")
bill = SequentialBill(id=1)
bill.make_random_position(dim=POLICY_SPACE_DIM)
print(f"   Bill position: {bill.position if hasattr(bill, 'position') else 'Random 3D'}")

# Step 4: Run simulation
print(f"\n[4] Running Monte Carlo simulation...")
engine_params = Session(
    n=500,  # Number of simulation iterations
    seed=SEED,
    bill=bill,
    description="Simple Congress Voting Simulation",
    congress_model=congress
)

engine = SequentialMonteCarlo(session_params=engine_params)
engine.run_simulation()

# Step 5: Print results
print(f"\n[5] Simulation Results:")
print("=" * 60)
print(engine)
print("=" * 60)
print("\nSimulation completed successfully!")
