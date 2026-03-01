from policyflux.utils.reports.bar_charts import craft_a_bar
from policyflux.utils.reports.pie_charts import bake_a_pie


def test_craft_a_bar_calls_pyplot_with_expected_arguments(monkeypatch) -> None:
    calls: dict[str, tuple] = {}

    monkeypatch.setattr(
        "policyflux.utils.reports.bar_charts.plt.figure",
        lambda *args, **kwargs: calls.__setitem__("figure", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.bar_charts.plt.bar",
        lambda *args, **kwargs: calls.__setitem__("bar", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.bar_charts.plt.title",
        lambda *args, **kwargs: calls.__setitem__("title", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.bar_charts.plt.xlabel",
        lambda *args, **kwargs: calls.__setitem__("xlabel", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.bar_charts.plt.ylabel",
        lambda *args, **kwargs: calls.__setitem__("ylabel", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.bar_charts.plt.xticks",
        lambda *args, **kwargs: calls.__setitem__("xticks", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.bar_charts.plt.tight_layout",
        lambda *args, **kwargs: calls.__setitem__("tight_layout", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.bar_charts.plt.show",
        lambda *args, **kwargs: calls.__setitem__("show", (args, kwargs)),
    )

    craft_a_bar(
        data=[3, 7],
        labels=["yes", "no"],
        title="Votes",
        xlabel="Type",
        ylabel="Count",
    )

    assert calls["figure"][1] == {"figsize": (10, 6)}
    assert calls["bar"][0] == (["yes", "no"], [3, 7])
    assert calls["bar"][1] == {"color": "skyblue"}
    assert calls["title"][0] == ("Votes",)
    assert calls["xlabel"][0] == ("Type",)
    assert calls["ylabel"][0] == ("Count",)
    assert calls["xticks"][1] == {"rotation": 45}
    assert "tight_layout" in calls
    assert "show" in calls


def test_bake_a_pie_calls_pyplot_with_expected_arguments(monkeypatch) -> None:
    calls: dict[str, tuple] = {}

    monkeypatch.setattr(
        "policyflux.utils.reports.pie_charts.plt.figure",
        lambda *args, **kwargs: calls.__setitem__("figure", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.pie_charts.plt.pie",
        lambda *args, **kwargs: calls.__setitem__("pie", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.pie_charts.plt.title",
        lambda *args, **kwargs: calls.__setitem__("title", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.pie_charts.plt.axis",
        lambda *args, **kwargs: calls.__setitem__("axis", (args, kwargs)),
    )
    monkeypatch.setattr(
        "policyflux.utils.reports.pie_charts.plt.show",
        lambda *args, **kwargs: calls.__setitem__("show", (args, kwargs)),
    )

    bake_a_pie(data=[4, 6], labels=["for", "against"], title="Outcome")

    assert calls["figure"][1] == {"figsize": (8, 8)}
    assert calls["pie"][0] == ([4, 6],)
    assert calls["pie"][1] == {
        "labels": ["for", "against"],
        "autopct": "%1.1f%%",
        "startangle": 140,
    }
    assert calls["title"][0] == ("Outcome",)
    assert calls["axis"][0] == ("equal",)
    assert "show" in calls
