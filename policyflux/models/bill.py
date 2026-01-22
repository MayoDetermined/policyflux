from typing import Optional
from ..core.bill_template import Bill
from ..core.id_generator import get_id_generator

class SequentialBill(Bill):
    def __init__(self, id: Optional[int] = None, position: Optional[list] = None) -> None:
        if id is None:
            id = get_id_generator().generate_bill_id()
        super().__init__(id)
        self.position: list[float] = position if position is not None else []

        self.n_passed: int = 0
        self.n_failed: int = 0

    def record_pass(self) -> None:
        self.n_passed += 1

    def record_fail(self) -> None:
        self.n_failed += 1
    
    def make_report(self) -> str:
        """Generate a report about the bill."""
        report = f"Bill {self.id}\n"
        report += f"Position: {self.position}\n"
        report += f"Passed: {self.n_passed}, Failed: {self.n_failed}\n"
        return report