"""Tests for policyflux.toolbox.bill.SequentialBill."""

import pytest

import policyflux.pfrandom as pfrandom
from policyflux.toolbox.bill import SequentialBill


class TestSequentialBillConstruction:
    def test_auto_id(self) -> None:
        bill = SequentialBill()
        assert isinstance(bill.id, int)
        assert bill.id > 0

    def test_explicit_id(self) -> None:
        bill = SequentialBill(id=42)
        assert bill.id == 42

    def test_default_position_empty(self) -> None:
        bill = SequentialBill()
        assert bill.position == []

    def test_explicit_position(self) -> None:
        bill = SequentialBill(position=[0.1, 0.2, 0.3])
        assert bill.position == [0.1, 0.2, 0.3]

    def test_initial_counters(self) -> None:
        bill = SequentialBill()
        assert bill.n_passed == 0
        assert bill.n_failed == 0

    def test_default_flags(self) -> None:
        bill = SequentialBill()
        assert bill.is_government_bill is False
        assert bill.is_confidence_vote is False


class TestSequentialBillPosition:
    def test_make_random_position_correct_dimension(self) -> None:
        pfrandom.set_seed(42)
        bill = SequentialBill()
        bill.make_random_position(dim=5)
        assert len(bill.position) == 5
        assert all(0.0 <= v <= 1.0 for v in bill.position)

    def test_make_random_position_dimension_one(self) -> None:
        pfrandom.set_seed(42)
        bill = SequentialBill()
        bill.make_random_position(dim=1)
        assert len(bill.position) == 1

    def test_make_random_position_overwrites_previous(self) -> None:
        pfrandom.set_seed(42)
        bill = SequentialBill(position=[0.5])
        bill.make_random_position(dim=3)
        assert len(bill.position) == 3


class TestSequentialBillRecording:
    def test_record_pass(self) -> None:
        bill = SequentialBill()
        bill.record_pass()
        assert bill.n_passed == 1
        bill.record_pass()
        assert bill.n_passed == 2
        assert bill.n_failed == 0

    def test_record_fail(self) -> None:
        bill = SequentialBill()
        bill.record_fail()
        assert bill.n_failed == 1
        bill.record_fail()
        assert bill.n_failed == 2
        assert bill.n_passed == 0

    def test_record_pass_and_fail(self) -> None:
        bill = SequentialBill()
        bill.record_pass()
        bill.record_fail()
        bill.record_pass()
        assert bill.n_passed == 2
        assert bill.n_failed == 1


class TestSequentialBillReport:
    def test_make_report_returns_string(self) -> None:
        bill = SequentialBill(id=10, position=[0.5, 0.5])
        report = bill.make_report()
        assert isinstance(report, str)

    def test_make_report_contains_bill_info(self) -> None:
        bill = SequentialBill(id=10, position=[0.5, 0.5])
        bill.record_pass()
        bill.record_pass()
        bill.record_fail()
        report = bill.make_report()
        assert "10" in report
        assert "Passed" in report or "2" in report
        assert "Failed" in report or "1" in report
