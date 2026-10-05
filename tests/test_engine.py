"""Unit tests for the twin: queue model, BOM explosion, simulator, optimizer, forecasting models."""
import numpy as np
import pytest

from sc import models, twin, world


@pytest.fixture(scope="module")
def small():
    w = world.build(0.1, seed=4)
    return w, twin.from_world(w)


def test_port_queue_model():
    assert twin.port_wait(-1, 0.7, 21) == 0                                   # reached the port before the cut
    assert twin.port_wait(5, 0.1, 21) == 0                                    # 90% capacity still covers 80% utilisation: no queue
    # 70% cut: backlog grows 0.5/day and is served at 0.3/day. A shipment arriving on day 21 finds 10.5 days of work, served at full speed.
    assert twin.port_wait(21, 0.7, 21) == pytest.approx(10.5)
    assert twin.port_wait(10, 0.7, 21) == pytest.approx(11 + (5 - 3.3))       # 3.3 cleared before capacity returns, 1.7 after
    assert twin.port_wait(21 + 10.5 / 0.2, 0.7, 21) == pytest.approx(0, abs=1e-9)        # queue drained at 0.2/day
    w = twin.port_wait(np.arange(0, 80), 0.7, 21)
    assert w.argmax() == 8 and w.max() == pytest.approx(13.1, abs=0.1) and (np.diff(w[25:]) <= 1e-9).all()
    assert twin.port_wait(10, np.array([0.5, 0.9]), 21)[1] > twin.port_wait(10, np.array([0.5, 0.9]), 21)[0]


def test_bom_is_exploded_through_sub_assemblies(small):
    w, tw = small
    pl = next(iter(tw.plants.values()))
    part = {p[0]: p for p in w["parts"]}
    kids = {}
    for parent, child, q in w["bom"]:
        kids.setdefault(parent, []).append((child, q))
    fg = pl.fg_ids[0]
    want = {}
    for child, q in kids[fg]:                                                  # recompute by hand: direct components + components of sub-assemblies
        if part[child][5] == "component":
            want[child] = want.get(child, 0) + q
        else:
            for c2, q2 in kids[child]:
                want[c2] = want.get(c2, 0) + q * q2
    got = {c: v for c, v in zip(pl.comp_ids, pl.q[0]) if v}
    assert got == pytest.approx(want) and any(part[c][5] == "subassembly" for c, _ in kids[fg])


def test_undisturbed_network_loses_nothing_and_a_cut_does(small):
    _, tw = small
    calm = twin.totals(twin.simulate(tw, twin.baseline_arrivals(tw)))
    assert calm["lost_revenue"][0] == 0 and calm["service_level"][0] == 1
    port = next(k for k, v in tw.ports.items() if v == "Shanghai")
    hits = twin.affected(tw, port, 0.7, 21)
    assert hits and all(tw.ships[k]["origin"] == port and tw.ships[k]["etd"] >= 0 for k, _ in hits)
    hurt = twin.totals(twin.simulate(tw, twin.expected_arrivals(tw, hits)))
    assert hurt["lost_revenue"][0] > 0 and hurt["stockout_component_days"][0] > 0
    assert twin.affected(tw, port, 0.1, 21) == []                               # a cut the port can absorb delays nothing


def test_optimizer_beats_doing_nothing_and_flying_everything(small):
    _, tw = small
    port = next(k for k, v in tw.ports.items() if v == "Shanghai")
    hits = twin.affected(tw, port, 0.7, 21)
    net = {}
    plans = {"nothing": {"expedite": [], "extra": {}, "cost": twin.plan_cost(tw, [], [])}, "naive": twin.naive_plan(tw, hits), "opt": twin.optimize(tw, port, hits)}
    for name, p in plans.items():
        t = twin.totals(twin.simulate(tw, twin.expected_arrivals(tw, hits, p), p["extra"]))
        net[name] = float(t["lost_margin"][0]) + sum(p["cost"].values())
    assert net["opt"] < net["nothing"] and net["opt"] < net["naive"]
    opt = plans["opt"]
    assert opt["solver"]["status"] == "optimal"
    flown = sum(tw.ships[k]["value"] for k in opt["expedite"])
    assert flown <= twin.AIR_VALUE_CAP * sum(tw.ships[k]["value"] for k, _ in hits) + 1e-6
    for a in opt["actions"]:                                                    # a second source never sits behind the disrupted port
        assert a["qty"] > 0 and a["first_day"] >= 0


def test_monte_carlo_is_reproducible_and_wider_than_the_expected_case(small):
    _, tw = small
    port = next(k for k, v in tw.ports.items() if v == "Shanghai")
    a, sev, dur = twin.monte_carlo(tw, port, 0.7, 21, runs=200, seed=0)
    b, _, _ = twin.monte_carlo(tw, port, 0.7, 21, runs=200, seed=0)
    assert (a["lost_revenue"] == b["lost_revenue"]).all() and a["lost_revenue"].shape == (200,)
    assert np.percentile(a["lost_revenue"], 90) > np.percentile(a["lost_revenue"], 10)
    assert 0.5 < sev.mean() < 0.9 and 15 < np.median(dur) < 28


def test_forecast_models_beat_their_baselines(small):
    w, _ = small
    _, m = models.train_eta(w["history"])
    assert m["model"]["mae_days"] < m["baseline"]["mae_days"] and 0.7 < m["model"]["interval_80_coverage"] < 0.9
    _, r, score, _ = models.train_risk(w["history"])
    assert r["model"]["brier"] < r["baseline"]["brier"] and r["model"]["auc"] > 0.7 and all(0 <= s <= 1 for s in score.values())
