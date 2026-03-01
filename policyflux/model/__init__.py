"""PolicyFlux Model API - TensorFlow-style model building.

Two complementary interfaces are provided:

Sequential API
--------------
Stack layers one at a time with :meth:`~Sequential.add` or the ``|``
pipe operator::

    from policyflux.model import Sequential, layers as L

    model = (
        Sequential(num_actors=100, policy_dim=4)
        | L.IdealPoint()
        | L.PublicOpinion(support=0.6)
        | L.PartyDiscipline(strength=0.5)
    )
    model.compile(executive='parliamentary', pm_strength=0.65)
    results = model.run(iterations=300, seed=42)
    model.summary()

Functional API
--------------
Build a computation graph with :class:`Input` nodes and callable layer
specs, then wrap with :class:`Model`::

    from policyflux.model import Model, Input, layers as L

    bill = Input(policy_dim=4, num_actors=100)
    x    = L.IdealPoint()(bill)
    x    = L.PublicOpinion(support=0.6)(x)
    x    = L.PartyDiscipline(strength=0.5)(x)

    model = Model(inputs=bill, outputs=x)
    model.compile(executive='parliamentary', pm_strength=0.65)
    results = model.run(iterations=300, seed=42)

Available layer specs (``policyflux.model.layers``)
----------------------------------------------------
- :class:`~policyflux.model.layers.IdealPoint`
- :class:`~policyflux.model.layers.PublicOpinion`
- :class:`~policyflux.model.layers.Lobbying`
- :class:`~policyflux.model.layers.MediaPressure`
- :class:`~policyflux.model.layers.PartyDiscipline`
- :class:`~policyflux.model.layers.GovernmentAgenda`

``compile()`` parameters
-------------------------
+---------------------+------------------------------------------------------+
| Parameter           | Description                                          |
+=====================+======================================================+
| executive           | ``'presidential'``, ``'parliamentary'``,             |
|                     | ``'semi_presidential'`` (and aliases)                |
+---------------------+------------------------------------------------------+
| aggregation         | ``'sequential'`` *(default)*, ``'average'``,         |
|                     | ``'weighted'``, ``'multiplicative'``                 |
+---------------------+------------------------------------------------------+
| approval            | Executive approval rating ``[0, 1]``                 |
+---------------------+------------------------------------------------------+
| veto_override       | Veto override threshold (presidential, default 2/3)  |
+---------------------+------------------------------------------------------+
| pm_strength         | PM party strength (parliamentary / semi, ``[0, 1]``) |
+---------------------+------------------------------------------------------+
| confidence_threshold| Confidence-vote threshold (parliamentary)            |
+---------------------+------------------------------------------------------+
| government_bill_rate| Fraction of bills that are government bills          |
+---------------------+------------------------------------------------------+
| n_lobbyists         | Number of lobbyist special actors                    |
+---------------------+------------------------------------------------------+
| n_whips             | Number of party-whip special actors                  |
+---------------------+------------------------------------------------------+
"""

from . import layers
from ._functional_model import Model
from ._node import Input
from ._sequential import Sequential

__all__ = [
    "Input",
    "Model",
    "Sequential",
    "layers",
]
