"""Public API (modular monolith). Blueprint services map to: digital-twin-service + risk-engine + scenario-engine +
optimization-service (sc/twin.py), eta-forecast-service (sc/models.py), geospatial + analytics (this file).

    uvicorn sc.api:app          python -m core.jobs sc.api      # the worker
"""
import datetime
import uuid

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import models, twin
from .world import CARRIERS, PLANTS, PORTS

READ = {"twin:read", "jobs:read"}
PERMISSIONS = {"viewer": READ, "planner": READ | {"scenarios:write", "scenarios:run", "plans:write"}, "admin": READ | {"scenarios:write", "scenarios:run", "plans:write", "audit:read"}}
app = create_app("supply-chain-twin", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
_twins, _models, _maps = {}, {}, {}


def get_twin(c, ctx):
    """The tenant's network in memory, built once per process from the system of record (RLS scopes every query)."""
    if ctx.tenant_id not in _twins:
        now = c.execute("SELECT max(i.observed_at) AS t FROM inventory_positions i JOIN facilities f ON f.id = i.facility_id").fetchone()["t"]
        q = lambda sql: c.execute(sql).fetchall()      # noqa: E731
        _twins[ctx.tenant_id] = twin.build(
            now, q("SELECT id, facility_type, name FROM facilities"), q("SELECT id, sku, unit_cost, part_type, home_facility_id, weekly_demand, unit_price FROM parts"),
            q("SELECT b.* FROM bom_edges b JOIN parts p ON p.id = b.parent_part_id"),
            q("SELECT sp.* FROM supplier_parts sp JOIN suppliers s ON s.id = sp.supplier_id"), q("SELECT id, name, origin_facility_id FROM suppliers"),
            q("""SELECT sh.id, sh.destination_id, l.part_id, l.qty, sh.origin_id, sh.etd, sh.eta, po.supplier_id, sh.mode, sh.status, sh.carrier
                   FROM shipments sh JOIN purchase_orders po ON po.id = sh.po_id JOIN suppliers s ON s.id = po.supplier_id JOIN po_lines l ON l.po_id = po.id
                  WHERE sh.status <> 'delivered'"""),
            q("SELECT i.facility_id, i.part_id, i.on_hand FROM inventory_positions i JOIN facilities f ON f.id = i.facility_id"))
    return _twins[ctx.tenant_id]


def model(name):
    if name not in _models:
        with db.tx() as c:
            row = c.execute("SELECT version, artifact FROM model_registry WHERE name = %s AND state = 'production'", [name]).fetchone()
        _models[name] = (row["version"], models.loads(row["artifact"]))
    return _models[name]


def get_scenario(c, sid):
    s = c.execute("SELECT * FROM scenarios WHERE id = %s", [sid]).fetchone()
    if not s:
        raise Problem(404, "scenario_not_found")
    return s


def scenario_hits(tw, s):
    a = s["assumptions"]
    return uuid.UUID(a["facility_id"]), twin.affected(tw, uuid.UUID(a["facility_id"]), a["severity"], a["duration_days"])


def money(x):
    return round(float(x), 2)


# ---------------------------------------------------------------- geospatial: map state

@app.get("/v1/map/state", tags=["map"], summary="Current network state for map rendering")
def map_state(scenario: uuid.UUID | None = None, ctx: Ctx = Depends(auth("twin:read"))):
    with db.tx(ctx.tenant_id) as c:
        disrupted = get_scenario(c, scenario)["assumptions"]["facility_id"] if scenario else None
        if ctx.tenant_id not in _maps:      # the network changes only when the twin is rebuilt, so the payload is built once
            _maps[ctx.tenant_id] = _map_payload(c, get_twin(c, ctx))
    base = _maps[ctx.tenant_id]
    return {**base, "disrupted_facility_id": disrupted, "facilities": [{**f, "disrupted": f["id"] == disrupted} for f in base["facilities"]]}


def _map_payload(c, tw):
    fac = c.execute("SELECT id, facility_type, name, lat, lon FROM facilities ORDER BY facility_type, name").fetchall()
    counts = c.execute("""SELECT (SELECT count(*) FROM suppliers) AS suppliers, (SELECT count(*) FROM parts) AS parts,
                                 (SELECT count(*) FROM (SELECT sp.part_id FROM supplier_parts sp JOIN suppliers s ON s.id = sp.supplier_id GROUP BY 1 HAVING count(*) = 1) x) AS single_sourced""").fetchone()
    pos = {f["id"]: (f["lat"], f["lon"]) for f in fac}
    moving = [s for s in tw.ships if s["status"] == "in_transit" and s["origin"] and s["eta"] > 0]
    step = max(1, len(moving) // 500)                 # the map shows a sample; the KPIs count everything
    dots = [{"id": s["id"], "from": pos[s["origin"]], "to": pos[s["plant"]], "progress": round(min(1, -s["etd"] / (s["eta"] - s["etd"])), 3), "origin_id": s["origin"]} for s in moving[::step]]
    flows = {}
    for s in tw.ships:
        if s["origin"]:
            k = (s["origin"], s["plant"])
            flows[k] = flows.get(k, 0) + s["value"]
    return jsonable_encoder({
        "as_of": tw.now, "engine": twin.VERSION, "facilities": fac, "in_transit_sample": dots,
        "lanes": [{"from": pos[a], "to": pos[b], "origin_id": a, "value": money(v)} for (a, b), v in sorted(flows.items(), key=lambda kv: -kv[1])],
        "kpis": {**counts, "facilities": len(fac), "in_transit": sum(s["status"] == "in_transit" for s in tw.ships), "planned": sum(s["status"] == "planned" for s in tw.ships),
                 "inbound_value": money(sum(s["value"] for s in tw.ships)), "finished_products": sum(len(p.fg_ids) for p in tw.plants.values()),
                 "weekly_revenue": money(sum(float(p.demand @ p.price) * 7 for p in tw.plants.values()))}})


# ---------------------------------------------------------------- scenario-engine

class ScenarioIn(BaseModel):
    name: str = Field(min_length=3, max_length=120)
    event_type: str = Field(pattern="^port_capacity_reduction$", description="The one disruption type modelled in this build")
    facility: str = Field(description="Port name, e.g. Shanghai")
    severity: float = Field(gt=0, lt=1, description="Share of port capacity lost")
    duration_days: int = Field(ge=1, le=60)


@app.post("/v1/scenarios", status_code=201, tags=["scenarios"], summary="Create a disruption scenario")
def create_scenario(body: ScenarioIn, ctx: Ctx = Depends(auth("scenarios:write")), idem: str | None = IdemKey):
    def work(c):
        f = c.execute("SELECT id, name, lat, lon FROM facilities WHERE name = %s AND facility_type = 'port'", [body.facility]).fetchone()
        if not f:
            raise Problem(404, "facility_not_found", f"no port named '{body.facility}'")
        tw = get_twin(c, ctx)
        sid = uuid.uuid7()
        a = {"event_type": body.event_type, "facility_id": str(f["id"]), "facility": f["name"], "severity": body.severity, "duration_days": body.duration_days,
             "port_utilization": 0.8, "engine": twin.VERSION}
        c.execute("INSERT INTO scenarios (id, tenant_id, name, assumptions, created_by) VALUES (%s,%s,%s,%s,%s)", [sid, ctx.tenant_id, body.name, Jsonb(a), ctx.actor_id])
        c.execute("INSERT INTO disruption_events (id, event_type, severity, lat, lon, starts_at, ends_at, source, payload, tenant_id) VALUES (%s,%s,%s,%s,%s,%s,%s,'what-if',%s,%s)",
                  [uuid.uuid7(), body.event_type, body.severity, f["lat"], f["lon"], tw.now, tw.now + datetime.timedelta(days=body.duration_days), Jsonb({"scenario_id": str(sid)}), ctx.tenant_id])
        audit.record(c, ctx, "scenario.created", "scenario", sid, a)
        return 201, {"id": sid, "name": body.name, "assumptions": a}
    return run(ctx, idem, body, work)


@app.get("/v1/exposure", tags=["scenarios"], summary="Impacted shipments, components, finished products, facilities and revenue (expected case)")
def exposure(event: uuid.UUID, ctx: Ctx = Depends(auth("twin:read"))):
    with db.tx(ctx.tenant_id) as c:
        s, tw = get_scenario(c, event), get_twin(c, ctx)
        single = {r["part_id"] for r in c.execute("SELECT sp.part_id FROM supplier_parts sp JOIN suppliers s ON s.id = sp.supplier_id GROUP BY 1 HAVING count(*) = 1")}
        sup = {r["id"]: r["name"] for r in c.execute("SELECT id, name FROM suppliers")}
    a = s["assumptions"]
    port, hits = scenario_hits(tw, s)
    sim = twin.simulate(tw, twin.expected_arrivals(tw, hits), trace=True)
    base = twin.simulate(tw, twin.baseline_arrivals(tw), trace=True)
    hit_parts = {tw.ships[k]["part"] for k, _ in hits}
    products, plants = [], []
    for pid, pl in tw.plants.items():
        v = sim[pid]
        cols = [j for j, cid in enumerate(pl.comp_ids) if cid in hit_parts]
        depends = (pl.q[:, cols] > 0).any(axis=1) if cols else np.zeros(len(pl.fg_ids), bool)      # BOM propagation: products that contain a delayed component
        for f in np.flatnonzero(depends):
            products.append({"sku": pl.fg_sku[f], "plant": pl.name, "delayed_components": int((pl.q[f, cols] > 0).sum()), "weekly_revenue": money(pl.demand[f] * 7 * pl.price[f]),
                             "lost_units": round(float(v["lost_sales"][0, f]), 1), "revenue_at_risk": money(v["lost_sales"][0, f] * pl.price[f])})
        short = np.flatnonzero(v["short_days"][0] > 0)
        worst = sorted(short, key=lambda j: -v["short_days"][0, j])[:3]
        plants.append({"facility_id": pid, "plant": pl.name, "delayed_shipments": sum(tw.ships[k]["plant"] == pid for k, _ in hits),
                       "first_stockout_day": None if np.isinf(v["first_stockout"][0]) else int(v["first_stockout"][0]), "components_short": len(short),
                       "lost_revenue": money(v["lost_revenue"][0]),
                       "depletion": [{"sku": pl.comp_sku[j], "single_sourced": pl.comp_ids[j] in single, "days_short": int(v["short_days"][0, j]),
                                      "cover_days": [round(float(x), 2) for x in v["cover"][:, j]], "cover_days_undisturbed": [round(float(x), 2) for x in base[pid]["cover"][:, j]]} for j in worst]})
    t = twin.totals(sim)
    delays = [d for _, d in hits]
    top = sorted(hits, key=lambda h: -h[1] * tw.ships[h[0]]["value"])[:10]
    return jsonable_encoder({
        "scenario_id": event, "assumptions": a, "engine": twin.VERSION,
        "shipments": {"affected": len(hits), "value": money(sum(tw.ships[k]["value"] for k, _ in hits)), "mean_delay_days": round(float(np.mean(delays)), 1) if delays else 0,
                      "max_delay_days": round(max(delays), 1) if delays else 0,
                      "delay_by_departure_day": [[d, round(float(twin.port_wait(d, a["severity"], a["duration_days"])), 2)] for d in range(0, 85)],
                      "top": [{"id": tw.ships[k]["id"], "supplier": sup[tw.ships[k]["supplier"]], "supplier_id": tw.ships[k]["supplier"], "carrier": tw.ships[k]["carrier"],
                               "departs_in_days": round(tw.ships[k]["etd"], 1), "delay_days": round(d, 1), "value": money(tw.ships[k]["value"])} for k, d in top]},
        "components": {"delayed": len(hit_parts), "single_sourced": len(hit_parts & single)},
        "finished_products": {"dependent": len(products), "losing_sales": sum(p["lost_units"] > 0 for p in products),
                              "top": sorted(products, key=lambda p: -p["revenue_at_risk"])[:10]},
        "facilities": sorted(plants, key=lambda p: -p["lost_revenue"]),
        "expected": {"lost_revenue": money(t["lost_revenue"][0]), "lost_margin": money(t["lost_margin"][0]), "service_level": round(float(t["service_level"][0]), 5),
                     "stockout_component_days": int(t["stockout_component_days"][0])}})


def _dist(x):
    q = np.percentile(x, [10, 50, 90])
    return {"p10": money(q[0]), "p50": money(q[1]), "p90": money(q[2]), "mean": money(np.mean(x))}


class RunIn(BaseModel):
    runs: int = Field(1000, ge=100, le=5000)


@app.post("/v1/scenarios/{sid}:run", status_code=202, tags=["scenarios"], summary="Run the Monte Carlo impact simulation (job)")
def run_scenario(sid: uuid.UUID, body: RunIn, ctx: Ctx = Depends(auth("scenarios:run")), idem: str | None = IdemKey):
    def work(c):
        get_scenario(c, sid)
        return 202, jobs.enqueue(c, ctx, "scenario.run", {"scenario_id": str(sid), "runs": body.runs})
    return run(ctx, idem, body, work)


@jobs.handler("scenario.run")
def scenario_job(c, job):
    ctx = Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")
    s, tw = get_scenario(c, job["payload"]["scenario_id"]), get_twin(c, ctx)
    a, runs = s["assumptions"], job["payload"]["runs"]
    port = uuid.UUID(a["facility_id"])
    hit, sev, dur = twin.monte_carlo(tw, port, a["severity"], a["duration_days"], runs=runs, seed=0)
    calm, _, _ = twin.monte_carlo(tw, port, 0.0, a["duration_days"], runs=runs, seed=0)          # same random transit noise, no port cut
    extra = hit["lost_revenue"] - calm["lost_revenue"]
    edges = np.linspace(0, float(np.percentile(hit["lost_revenue"], 98)) or 1, 25)
    out = {"runs": runs, "engine": twin.VERSION, "sampled": {"severity_p10_p90": [round(float(x), 3) for x in np.percentile(sev, [10, 90])],
                                                              "duration_days_p10_p90": [round(float(x), 1) for x in np.percentile(dur, [10, 90])], "transit_noise_sd_days": 1.5},
           "lost_revenue": _dist(hit["lost_revenue"]), "lost_revenue_without_disruption": _dist(calm["lost_revenue"]), "lost_revenue_caused_by_disruption": _dist(extra),
           "service_level": {"p10": round(float(np.percentile(hit["service_level"], 10)), 5), "p50": round(float(np.percentile(hit["service_level"], 50)), 5)},
           "probability_disruption_costs_over_1m": round(float((extra > 1e6).mean()), 3),
           "histogram": {"edges": [money(e) for e in edges], "disrupted": np.histogram(hit["lost_revenue"], edges)[0].tolist(), "undisturbed": np.histogram(calm["lost_revenue"], edges)[0].tolist()}}
    c.execute("DELETE FROM scenario_results WHERE scenario_id = %s AND metric LIKE 'mc_%%'", [s["id"]])
    db.many(c, "INSERT INTO scenario_results (id, scenario_id, metric, entity_ref, value, detail) VALUES (%s,%s,%s,'network',%s,%s)",
            [(uuid.uuid7(), s["id"], f"mc_lost_revenue_{k}", v, Jsonb({"runs": runs})) for k, v in out["lost_revenue"].items()])
    audit.record(c, ctx, "scenario.simulated", "scenario", s["id"], {"runs": runs, "engine": twin.VERSION, "lost_revenue_p50": out["lost_revenue"]["p50"]})
    return out


# ---------------------------------------------------------------- optimization-service

class PlanIn(BaseModel):
    scenario_id: uuid.UUID
    runs: int = Field(500, ge=100, le=2000)


@app.post("/v1/recovery-plans", status_code=202, tags=["recovery"], summary="Optimize sourcing / airfreight / reserve response and compare with doing nothing (job)")
def recovery_plan(body: PlanIn, ctx: Ctx = Depends(auth("plans:write")), idem: str | None = IdemKey):
    def work(c):
        get_scenario(c, body.scenario_id)
        return 202, jobs.enqueue(c, ctx, "recovery.plan", {"scenario_id": str(body.scenario_id), "runs": body.runs})
    return run(ctx, idem, body, work)


@jobs.handler("recovery.plan")
def recovery_job(c, job):
    ctx = Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")
    s, tw = get_scenario(c, job["payload"]["scenario_id"]), get_twin(c, ctx)
    a, runs = s["assumptions"], job["payload"]["runs"]
    port, hits = scenario_hits(tw, s)
    plans = {"do_nothing": {"expedite": [], "extra": {}, "actions": [], "cost": twin.plan_cost(tw, [], [])}, "expedite_most_delayed": twin.naive_plan(tw, hits),
             "optimized": twin.optimize(tw, port, hits)}
    rows = []
    for name, p in plans.items():
        t = twin.totals(twin.simulate(tw, twin.expected_arrivals(tw, hits, p), p["extra"]))
        mc, _, _ = twin.monte_carlo(tw, port, a["severity"], a["duration_days"], plan=p, runs=runs, seed=1)
        cost = sum(p["cost"].values())
        rows.append({"plan": name, "shipments_flown": len(p["expedite"]), "alternate_buys": sum(x["type"] == "alternate_supplier" for x in p["actions"]),
                     "warehouse_transfers": sum(x["type"] == "warehouse_transfer" for x in p["actions"]),
                     "freight_cost": money(p["cost"]["airfreight"]), "other_cost": money(p["cost"]["alternate_supply"] + p["cost"]["transfers"]),
                     "lost_revenue": money(t["lost_revenue"][0]), "lost_margin": money(t["lost_margin"][0]), "service_level": round(float(t["service_level"][0]), 5),
                     "net_cost": money(t["lost_margin"][0] + cost), "monte_carlo_lost_revenue": _dist(mc["lost_revenue"]),
                     "monte_carlo_net_cost_p50": money(float(np.percentile(mc["lost_margin"], 50)) + cost)})
    opt = plans["optimized"]
    flown = sorted(opt["expedite"], key=lambda k: -tw.ships[k]["value"])
    out = {"scenario_id": str(s["id"]), "engine": twin.VERSION, "solver": opt["solver"], "comparison": rows,
           "assumptions": {"air_days": twin.AIR_DAYS, "air_cost": f"${twin.AIR_FIXED:,.0f} + {twin.AIR_RATE:.0%} of shipment value", "air_value_cap_share": twin.AIR_VALUE_CAP,
                           "alternate_premium": twin.ALT_PREMIUM, "transfer_cost": twin.TRANSFER_COST, "period_days": twin.BUCKET},
           "actions": {"fly": [{"shipment_id": str(tw.ships[k]["id"]), "value": money(tw.ships[k]["value"]), "delay_avoided_days": round(dict(hits)[k], 1),
                                "cost": money(twin.AIR_FIXED + twin.AIR_RATE * tw.ships[k]["value"])} for k in flown[:12]],
                       "source_and_transfer": sorted(opt["actions"], key=lambda x: -x["cost"])[:12]}}
    c.execute("DELETE FROM scenario_results WHERE scenario_id = %s AND metric LIKE 'plan_%%'", [s["id"]])
    db.many(c, "INSERT INTO scenario_results (id, scenario_id, metric, entity_ref, value, detail) VALUES (%s,%s,'plan_net_cost',%s,%s,%s)",
            [(uuid.uuid7(), s["id"], r["plan"], r["net_cost"], Jsonb(r)) for r in rows])
    audit.record(c, ctx, "recovery.planned", "scenario", s["id"], {"solver": opt["solver"]["status"], "engine": twin.VERSION, "net_cost_optimized": rows[2]["net_cost"], "net_cost_do_nothing": rows[0]["net_cost"]})
    return jsonable_encoder(out)


# ---------------------------------------------------------------- eta-forecast-service and supplier risk

@app.get("/v1/shipments/{shipment_id}/eta", tags=["forecast"], summary="Predicted arrival with an 80% interval, next to the promised date")
def shipment_eta(shipment_id: uuid.UUID, scenario: uuid.UUID | None = None, ctx: Ctx = Depends(auth("twin:read"))):
    with db.tx(ctx.tenant_id) as c:
        s = c.execute("""SELECT sh.*, o.name AS origin, d.name AS destination,
                                (SELECT congestion_index FROM port_congestion pc WHERE pc.facility_id = sh.origin_id AND pc.week_start <= sh.etd::date ORDER BY week_start DESC LIMIT 1) AS congestion
                           FROM shipments sh JOIN purchase_orders po ON po.id = sh.po_id JOIN suppliers su ON su.id = po.supplier_id
                           LEFT JOIN facilities o ON o.id = sh.origin_id JOIN facilities d ON d.id = sh.destination_id WHERE sh.id = %s""", [shipment_id]).fetchone()
        if not s:
            raise Problem(404, "shipment_not_found")
        wait = 0.0
        if scenario:
            sc, tw = get_scenario(c, scenario), get_twin(c, ctx)
            if str(s["origin_id"]) == sc["assumptions"]["facility_id"]:
                wait = float(twin.port_wait(max(0, (s["etd"] - tw.now).total_seconds() / 86400), sc["assumptions"]["severity"], sc["assumptions"]["duration_days"]))
    promised = (s["eta"] - s["etd"]).total_seconds() / 86400
    port = next((i for i, p in enumerate(PORTS) if p[0] == s["origin"]), -1)
    x = [models.eta_features(port, next(i for i, p in enumerate(PLANTS) if p[0] == s["destination"]), CARRIERS.index(s["carrier"]), s["etd"].month, float(s["congestion"] or 0), promised - 2)]
    version, m = model("eta-forecast")
    days = {q: float(m[q].predict(x)[0]) + wait for q in (0.1, 0.5, 0.9)}
    at = lambda d: s["etd"] + datetime.timedelta(days=d)      # noqa: E731
    return jsonable_encoder({"shipment_id": shipment_id, "status": s["status"], "lane": f"{s['origin'] or 'overland'} to {s['destination']}", "carrier": s["carrier"], "etd": s["etd"],
                             "promised_eta": s["eta"], "predicted_eta": at(days[0.5]), "interval_80": [at(days[0.1]), at(days[0.9])],
                             "predicted_transit_days": round(days[0.5], 1), "promised_transit_days": round(promised, 1), "scenario_port_wait_days": round(wait, 1),
                             "model_version": version, "evidence": {"port_congestion_index": float(s["congestion"] or 0), "month": s["etd"].month}})


@app.get("/v1/suppliers/{supplier_id}/risk", tags=["forecast"], summary="Supplier delay risk, its drivers and what depends on the supplier")
def supplier_risk(supplier_id: uuid.UUID, ctx: Ctx = Depends(auth("twin:read"))):
    with db.tx(ctx.tenant_id) as c:
        s = c.execute("SELECT s.*, f.name AS port FROM suppliers s LEFT JOIN facilities f ON f.id = s.origin_facility_id WHERE s.id = %s", [supplier_id]).fetchone()
        if not s:
            raise Problem(404, "supplier_not_found")
        hist = c.execute("""SELECT count(*) AS deliveries, round(avg((sh.delivered_at > sh.eta + interval '3 days')::int), 4) AS late_rate,
                                   round(avg(extract(epoch FROM sh.delivered_at - sh.eta) / 86400)::numeric, 2) AS mean_days_vs_promise
                              FROM shipments sh JOIN purchase_orders po ON po.id = sh.po_id WHERE po.supplier_id = %s AND sh.status = 'delivered'""", [supplier_id]).fetchone()
        mine = c.execute("""SELECT sp.part_id, p.sku, sp.allocation_pct, (SELECT count(*) FROM supplier_parts x WHERE x.part_id = sp.part_id) AS sources
                              FROM supplier_parts sp JOIN parts p ON p.id = sp.part_id WHERE sp.supplier_id = %s""", [supplier_id]).fetchall()
        tw = get_twin(c, ctx)
        overall = c.execute("SELECT round(avg(risk_score), 4) AS r FROM suppliers").fetchone()["r"]
    ids = {m["part_id"] for m in mine}
    dependents, revenue = [], 0.0
    for pl in tw.plants.values():
        cols = [j for j, cid in enumerate(pl.comp_ids) if cid in ids]
        for f in np.flatnonzero((pl.q[:, cols] > 0).any(axis=1)) if cols else []:
            dependents.append(pl.fg_sku[f])
            revenue += float(pl.demand[f] * 7 * pl.price[f])
    version, _ = model("supplier-delay")
    return jsonable_encoder({"supplier_id": supplier_id, "name": s["name"], "tier": s["tier"], "country": s["country"], "ships_through": s["port"],
                             "risk_score": s["risk_score"], "network_average": overall, "model_version": version,
                             "drivers": {**hist, "late_definition": "more than 3 days after the promised date"},
                             "dependency": {"components_supplied": len(mine), "single_sourced_components": [m["sku"] for m in mine if m["sources"] == 1][:20],
                                            "finished_products_depending": len(dependents), "weekly_revenue_depending": money(revenue)}})
