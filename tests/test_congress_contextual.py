import numpy as np
from policyflux.congress.actors import CongressMan
from policyflux.congress.congress import TheCongress
from policyflux.congress.law import Law
from policyflux.public_opinion.regime import PublicRegime


def make_sample_congress(n=6):
    actors = []
    for i in range(n):
        data = {
            "ideology_multidim": [np.random.uniform(-1, 1) for _ in range(3)],
            "loyalty": np.random.uniform(0.2, 0.9),
            "vulnerability": np.random.uniform(0.05, 0.3),
            "volatility": np.random.uniform(0.01, 0.2),
            "party": 'Republican' if i % 2 == 0 else 'Democratic',
        }
        actors.append(CongressMan(i, data))

    adj = np.random.rand(len(actors), len(actors)) * 0.2
    np.fill_diagonal(adj, 0.0)
    regime = PublicRegime(scenario='polarized')
    cong = TheCongress(actors, adj, regime)
    return cong


def test_contextual_influence_increases_intraparty_with_pressure():
    cong = make_sample_congress(8)
    law = Law.create_random(law_id=1, dim=3)

    W_low = cong.get_contextual_influence_matrix(law, regime_pressure=0.1)
    W_high = cong.get_contextual_influence_matrix(law, regime_pressure=0.9)

    party_matrix = cong._get_party_homophily_matrix()

    # compute mean intra-party weight
    intra_low = W_low[party_matrix == 1.0]
    intra_high = W_high[party_matrix == 1.0]

    # take mean, handle empty
    mean_low = float(intra_low.mean()) if intra_low.size > 0 else 0.0
    mean_high = float(intra_high.mean()) if intra_high.size > 0 else 0.0

    # High regime pressure should lead to increased intra-party weights
    assert mean_high >= mean_low

    # shape preserved and no NaNs
    assert W_low.shape == W_high.shape
    assert not np.isnan(W_low).any()
    assert not np.isnan(W_high).any()




