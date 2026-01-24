#!/usr/bin/env python3
"""
Prosty przykład użycia policyflux
Uruchom: python examples/simple_example.py
"""

import random
from policyflux.models import (
    SequentialCongressModel,
    SequentialVoter,
    SequentialBill,
    Session,
    SequentialMonteCarlo,
)
from policyflux.layers import IdealPointEncoder, PublicOpinionLayer

SEED = 123
NUM_ACTORS = 50
POLICY_DIM = 2
ITERATIONS = 300

random.seed(SEED)

def build_congress(num_actors: int, dim: int) -> SequentialCongressModel:
    congress = SequentialCongressModel(id=1)
    for i in range(1, num_actors + 1):
        actor = SequentialVoter(
            id=i,
            name=f"Rep-{i}",
            layers=[
                IdealPointEncoder(
                    id=1000 + i,
                    space=[random.random() for _ in range(dim)],
                    status_quo=[0.5] * dim,
                ),
                PublicOpinionLayer(id=2000 + i, support_level=random.random()),
            ],
        )
        congress.add_congressman(actor)
    congress.compile()
    return congress


def make_bill(dim: int) -> SequentialBill:
    bill = SequentialBill(id=1)
    bill.make_random_position(dim=dim)
    return bill


def run_example():
    congress = build_congress(NUM_ACTORS, POLICY_DIM)
    bill = make_bill(POLICY_DIM)

    session = Session(
        n=ITERATIONS,
        seed=SEED,
        bill=bill,
        description="Prosty przykład policyflux",
        congress_model=congress,
    )

    engine = SequentialMonteCarlo(session_params=session)
    engine.run_simulation()

    print("\nWynik symulacji:")
    print(engine)


if __name__ == "__main__":
    run_example()
