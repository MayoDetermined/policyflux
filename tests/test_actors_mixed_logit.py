import numpy as np
import pytest

import config
from congress.actors import CongressMan
from congress.law import Law


def make_actor() -> CongressMan:
    data = {
        "ideology_multidim": [0.2, -0.1, 0.5],
        "loyalty": 0.7,
        "vulnerability": 0.1,
        "volatility": 0.05,
        "party": "Democratic",
        "political_capital": 0.8,
        "presidential_support_score": 0.4,
    }
    return CongressMan(1, data)


def test_calculate_mixed_logit_prob_bounds():
    actor = make_actor()
    law = Law.create_random(law_id=42, dim=3)
    prob = actor.calculate_mixed_logit_prob(law, local_regime_pressure=0.3)
    assert 0.0 <= prob <= 1.0


def test_mixed_logit_incorporates_random_eta():
    actor_low = make_actor()
    actor_high = make_actor()
    actor_low.random_eta = -0.5
    actor_high.random_eta = 0.5
    law = Law.create_random(law_id=99, dim=3)

    prob_low = actor_low.calculate_mixed_logit_prob(law, local_regime_pressure=0.5)
    prob_high = actor_high.calculate_mixed_logit_prob(law, local_regime_pressure=0.5)

    assert prob_low != pytest.approx(prob_high)


def test_decide_vote_uses_mixed_logit_if_enabled(monkeypatch):
    actor = make_actor()
    law = Law.create_random(law_id=7, dim=3)

    called = {"hit": False}

    def stub(law_arg, pressure_arg):
        called["hit"] = True
        return 0.8

    monkeypatch.setattr(actor, "calculate_mixed_logit_prob", stub)
    monkeypatch.setattr(config, "USE_MIXED_LOGIT", True)
    monkeypatch.setattr(np.random, "normal", lambda loc, scale: 0.0)
    monkeypatch.setattr(np.random, "random", lambda: 0.0)

    vote = actor.decide_vote(law, influence_effect=0.0, regime_pressure=0.2)
    assert called["hit"]
    assert vote is True or vote is False
