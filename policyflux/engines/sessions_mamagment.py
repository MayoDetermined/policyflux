from dataclasses import dataclass
from ..core.bill_template import Bill
from ..core.congress_model_template import CongressModel

@dataclass(frozen=True)
class Session:
    n: int
    seed: int
    bill: Bill
    description: str
    congress_model: CongressModel