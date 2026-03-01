from policyflux.core.congress_model import CongressModel


class _DummyCongressman:
    def __init__(self, vote_value: bool) -> None:
        self._vote_value = vote_value

    def vote(self, bill: object) -> bool:
        return self._vote_value


class _TestCongress(CongressModel):
    def make_report(self) -> str:
        return "ok"


def test_add_pop_and_empty_pop() -> None:
    congress = _TestCongress(id=1)
    member = _DummyCongressman(True)

    congress.add_congressman(member)
    assert congress.pop_congressman() is member
    assert congress.pop_congressman() is None


def test_delete_congressman_returns_true_or_false() -> None:
    congress = _TestCongress(id=1)
    member_a = _DummyCongressman(True)
    member_b = _DummyCongressman(False)
    congress.add_congressman(member_a)

    assert congress.delete_congressman(member_a) is True
    assert congress.delete_congressman(member_b) is False


def test_cast_votes_counts_votes_for() -> None:
    congress = _TestCongress(id=1)
    congress.add_congressman(_DummyCongressman(True))
    congress.add_congressman(_DummyCongressman(False))
    congress.add_congressman(_DummyCongressman(True))

    votes_for = congress.cast_votes(bill=object())

    assert votes_for == 2


def test_make_report_implemented_by_subclass() -> None:
    congress = _TestCongress(id=7)
    assert congress.make_report() == "ok"
