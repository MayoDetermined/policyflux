from dataclasses import dataclass

from ..core.abstract_bill import Bill
from ..core.congress_model import CongressModel


@dataclass(frozen=True)
class Session:
    n: int
    seed: int
    bill: Bill
    description: str
    congress_model: CongressModel
