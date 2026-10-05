"""Integration, end-to-end and security tests against a real Postgres (network at one-fifth scale)."""
import pytest

from core import db, jobs, scenario

from .conftest import bearer

PLANNER, VIEWER, ADMIN, OTHER = bearer("acme", "planner"), bearer("acme", "viewer"), bearer("acme", "admin"), bearer("other", "planner")


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def test_seed_registers_models_with_their_baselines(world):
    assert world["metrics"]["eta"]["model"]["mae_days"] < world["metrics"]["eta"]["baseline"]["mae_days"]
    with db.tx() as c:
        assert {r["name"] for r in c.execute("SELECT name FROM model_registry")} == {"eta-forecast", "supplier-delay"}
        assert c.execute("SELECT count(*) AS n FROM suppliers WHERE risk_score IS NOT NULL").fetchone()["n"] > 50


def test_demo_journey_end_to_end(demo):
    m = demo["map"]
    assert m["kpis"]["facilities"] == 50 and len(m["in_transit_sample"]) > 100 and m["lanes"]
    e = demo["exp"]
    assert e["shipments"]["affected"] > 100 and 12 < e["shipments"]["max_delay_days"] < 14
    assert e["finished_products"]["dependent"] >= e["finished_products"]["losing_sales"] > 0 and e["expected"]["lost_revenue"] > 0
    worst = e["facilities"][0]["depletion"][0]
    assert min(worst["cover_days"]) == 0 and min(worst["cover_days_undisturbed"]) > 0          # runs out only when the port is cut
    assert demo["eta"]["scenario_port_wait_days"] > 5 and demo["eta"]["predicted_eta"] > demo["eta"]["promised_eta"]
    assert demo["eta"]["interval_80"][0] < demo["eta"]["predicted_eta"] < demo["eta"]["interval_80"][1]
    assert demo["sup"]["ships_through"] == "Shanghai" and demo["sup"]["dependency"]["finished_products_depending"] > 0
    mc = demo["mc"]["result"]
    assert mc["runs"] == 1000 and mc["lost_revenue"]["p10"] <= mc["lost_revenue"]["p50"] <= mc["lost_revenue"]["p90"]
    assert mc["lost_revenue_without_disruption"]["p50"] <= mc["lost_revenue"]["p50"] and sum(mc["histogram"]["disrupted"]) >= 950
    rows = {r["plan"]: r for r in demo["plan"]["result"]["comparison"]}
    assert rows["optimized"]["net_cost"] < rows["do_nothing"]["net_cost"] and rows["optimized"]["net_cost"] < rows["expedite_most_delayed"]["net_cost"]
    assert rows["optimized"]["lost_revenue"] < rows["do_nothing"]["lost_revenue"] and rows["do_nothing"]["freight_cost"] == 0
    assert demo["plan"]["result"]["solver"]["status"] == "optimal"
    assert demo["audit"]["chain_valid"] and [x["action"] for x in demo["audit"]["events"]][::-1] == ["scenario.created", "scenario.simulated", "recovery.planned"]
    with db.tx() as c:
        metrics = {r["metric"] for r in c.execute("SELECT DISTINCT metric FROM scenario_results")}
    assert {"mc_lost_revenue_p50", "plan_net_cost"} <= metrics


@pytest.mark.parametrize("headers,method,path,body,expected", [
    ({}, "GET", "/v1/map/state", None, 401),
    (VIEWER, "GET", "/v1/map/state", None, 200),
    (VIEWER, "POST", "/v1/scenarios", {"name": "x y z", "event_type": "port_capacity_reduction", "facility": "Shanghai", "severity": 0.5, "duration_days": 5}, 403),
    (VIEWER, "POST", "/v1/recovery-plans", {"scenario_id": "00000000-0000-0000-0000-000000000000"}, 403),
    (PLANNER, "GET", "/v1/audit", None, 403),
    (ADMIN, "GET", "/v1/audit", None, 200),
])
def test_authorization_matrix(client, headers, method, path, body, expected):
    assert client.request(method, path, headers=headers, json=body).status_code == expected


def test_tenant_isolation(client, demo):
    sid = demo["scn"]["id"]
    assert client.get(f"/v1/exposure?event={sid}", headers=OTHER).status_code == 404
    assert client.post(f"/v1/scenarios/{sid}:run", headers=OTHER, json={}).status_code == 404
    assert client.post("/v1/recovery-plans", headers=OTHER, json={"scenario_id": sid}).status_code == 404
    assert client.get(f"/v1/shipments/{demo['exp']['shipments']['top'][0]['id']}/eta", headers=OTHER).status_code == 404
    assert client.get(f"/v1/suppliers/{demo['exp']['shipments']['top'][0]['supplier_id']}/risk", headers=OTHER).status_code == 404
    theirs = client.get("/v1/map/state", headers=OTHER).json()
    assert theirs["kpis"]["suppliers"] == 40 and theirs["kpis"]["suppliers"] < demo["map"]["kpis"]["suppliers"]


def test_bad_input_is_a_problem_document(client):
    bad = {"name": "abc", "event_type": "port_capacity_reduction", "facility": "Atlantis", "severity": 0.5, "duration_days": 5}
    r = client.post("/v1/scenarios", headers=PLANNER, json=bad)
    assert r.status_code == 404 and r.json()["code"] == "facility_not_found" and r.headers["content-type"].startswith("application/problem+json")
    assert client.post("/v1/scenarios", headers=PLANNER, json={**bad, "facility": "Shanghai", "severity": 1.5}).status_code == 422
    assert client.post("/v1/scenarios", headers=PLANNER, json={**bad, "event_type": "earthquake"}).status_code == 422
    assert client.get("/v1/exposure?event=not-a-uuid", headers=PLANNER).status_code == 422


def test_scenario_creation_is_idempotent(client):
    body = {"name": "Busan test", "event_type": "port_capacity_reduction", "facility": "Busan", "severity": 0.6, "duration_days": 10}
    a = client.post("/v1/scenarios", headers={**PLANNER, "Idempotency-Key": "k1"}, json=body)
    b = client.post("/v1/scenarios", headers={**PLANNER, "Idempotency-Key": "k1"}, json=body)
    assert a.status_code == 201 and b.json()["id"] == a.json()["id"] and b.headers["Idempotent-Replay"] == "true"
    with db.tx() as c:
        assert c.execute("SELECT count(*) AS n FROM scenarios WHERE name = 'Busan test'").fetchone()["n"] == 1
