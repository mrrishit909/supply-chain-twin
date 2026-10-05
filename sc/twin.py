"""The digital twin: the network in memory, a port-queue delay model, a daily production simulator (one future or
a thousand), and the recovery optimizer. Pure numpy / OR-Tools over plain rows, so it runs without the database.

Simplifications that matter are marked `ponytail:` in place.
"""
import dataclasses
import datetime
import math
from collections import defaultdict

import numpy as np
from ortools.linear_solver import pywraplp

VERSION = "twin-1.0.0"
HORIZON, BUCKET = 84, 3                 # days simulated; optimizer period length in days
AIR_DAYS, TRANSFER_DAYS = 5, 3
ALT_PREMIUM, TRANSFER_COST, AIR_RATE, AIR_FIXED = 0.20, 0.03, 0.08, 1800.0     # share of unit cost / of value / of value / per shipment
AIR_VALUE_CAP = 0.35                    # at most this share of the delayed value can be flown


@dataclasses.dataclass
class Plant:
    id: object
    name: str
    fg_ids: list
    fg_sku: list
    demand: np.ndarray            # units per day
    price: np.ndarray
    cost: np.ndarray
    buffer: np.ndarray            # finished units already in the warehouses
    comp_ids: list
    comp_sku: list
    unit_cost: np.ndarray
    inv0: np.ndarray
    reserve: np.ndarray           # units held at a regional warehouse
    q: np.ndarray                 # (finished, component): component units per finished unit, BOM fully exploded


@dataclasses.dataclass
class Twin:
    now: datetime.datetime
    plants: dict
    ports: dict                   # facility id -> name
    ships: list                   # open + planned inbound shipments (dicts)
    alt: dict                     # (plant id, component id) -> [(supplier id, name, spare units per week, lead days, origin port id)]


def build(now, facilities, parts, bom, supplier_parts, suppliers, shipments, inventory):
    """Assemble the twin from table rows (dicts). Explodes the multi-level BOM down to purchased components."""
    kids = defaultdict(list)
    for b in bom:
        kids[b["parent_part_id"]].append((b["child_part_id"], float(b["qty"])))
    part = {p["id"]: p for p in parts}
    memo = {}

    def explode(pid):          # graph propagation: which purchased components, and how many, sit under this part
        if pid not in memo:
            out = defaultdict(float)
            for child, q in kids.get(pid, []):
                if part[child]["part_type"] == "component":
                    out[child] += q
                else:
                    for c, qc in explode(child).items():
                        out[c] += q * qc
            memo[pid] = out
        return memo[pid]

    plant_rows = {f["id"]: f for f in facilities if f["facility_type"] == "plant"}
    on_hand, reserve = {}, defaultdict(float)
    for i in inventory:
        if i["facility_id"] in plant_rows:
            on_hand[(i["facility_id"], i["part_id"])] = float(i["on_hand"])
        else:
            reserve[i["part_id"]] += float(i["on_hand"])
    plants = {}
    for pid, f in plant_rows.items():
        fgs = sorted((p for p in parts if p["part_type"] == "finished" and p["home_facility_id"] == pid), key=lambda p: p["sku"])
        need = {p["id"]: explode(p["id"]) for p in fgs}
        comps = sorted({c for n in need.values() for c in n}, key=lambda c: part[c]["sku"])
        ci = {c: j for j, c in enumerate(comps)}
        q = np.zeros((len(fgs), len(comps)))
        for i, p in enumerate(fgs):
            for c, v in need[p["id"]].items():
                q[i, ci[c]] = v
        plants[pid] = Plant(pid, f["name"], [p["id"] for p in fgs], [p["sku"] for p in fgs], np.array([float(p["weekly_demand"]) / 7 for p in fgs]),
                            np.array([float(p["unit_price"]) for p in fgs]), np.array([float(p["unit_cost"]) for p in fgs]),
                            np.array([on_hand.get((pid, p["id"]), 0.0) for p in fgs]), comps, [part[c]["sku"] for c in comps],
                            np.array([float(part[c]["unit_cost"]) for c in comps]), np.array([on_hand.get((pid, c), 0.0) for c in comps]),
                            np.array([reserve.get(c, 0.0) for c in comps]), q)
    sup = {s["id"]: s for s in suppliers}
    rel = lambda t: (t - now).total_seconds() / 86400      # noqa: E731
    ships = [{"id": s["id"], "plant": s["destination_id"], "part": s["part_id"], "qty": float(s["qty"]), "origin": s["origin_id"], "etd": rel(s["etd"]), "eta": rel(s["eta"]),
              "supplier": s["supplier_id"], "value": float(s["qty"]) * float(part[s["part_id"]]["unit_cost"]), "mode": s["mode"], "status": s["status"], "carrier": s["carrier"]}
             for s in shipments if s["destination_id"] in plants]
    usage = {(pid, c): float(u) for pid, pl in plants.items() for c, u in zip(pl.comp_ids, pl.demand @ pl.q)}
    alt = defaultdict(list)
    for sp in supplier_parts:
        home = part[sp["part_id"]]["home_facility_id"]
        if (home, sp["part_id"]) in usage:
            spare = float(sp["capacity_per_week"]) - usage[(home, sp["part_id"])] * 7 * float(sp["allocation_pct"]) / 100
            alt[(home, sp["part_id"])].append((sp["supplier_id"], sup[sp["supplier_id"]]["name"], max(0.0, spare), float(sp["lead_time_days"]), sup[sp["supplier_id"]]["origin_facility_id"]))
    return Twin(now, plants, {f["id"]: f["name"] for f in facilities if f["facility_type"] == "port"}, ships, dict(alt))


def port_wait(t, severity, duration, rho=0.8):
    """Days a shipment queues if it reaches the port t days after a capacity cut of `severity` lasting `duration` days.

    Fluid first-in-first-out queue: work arrives at rate rho (the port's normal utilisation), is served at
    (1 - severity) during the cut and at 1 afterwards. Vectorised: any argument may be an array.
    """
    t, severity, duration = np.broadcast_arrays(np.asarray(t, float), np.asarray(severity, float), np.asarray(duration, float))
    slow = 1 - severity
    grow = np.maximum(0, rho - slow)
    peak = grow * duration
    backlog = np.where(t < 0, 0, np.where(t <= duration, grow * t, np.maximum(0, peak - (1 - rho) * (t - duration))))
    left = np.maximum(0, duration - t) * slow                      # work the port can still clear before the cut ends
    return np.where(backlog <= left, backlog / np.maximum(slow, 1e-9), np.maximum(0, duration - t) + (backlog - left))


def affected(twin, port_id, severity, duration):
    """Inbound shipments that will load at the port while it is cut or still draining its queue, with expected delay."""
    out = []
    for i, s in enumerate(twin.ships):
        if s["origin"] == port_id and s["etd"] >= 0:
            d = float(port_wait(s["etd"], severity, duration))
            if d >= 0.5:
                out.append((i, d))
    return out


def _receipts(pl, ships, arrive, runs):
    """(runs, components, HORIZON) units arriving per day. `arrive` is (runs, n_ships) day numbers."""
    ci = {c: j for j, c in enumerate(pl.comp_ids)}
    r = np.zeros((runs, len(pl.comp_ids), HORIZON))
    for k, s in enumerate(ships):
        day = np.maximum(0, np.floor(arrive[:, k])).astype(int)      # overdue in-transit shipments land today
        ok = day < HORIZON
        np.add.at(r, (np.flatnonzero(ok), ci[s["part"]], day[ok]), s["qty"])
    return r


def simulate(twin, arrivals, extra=None, runs=1, trace=False):
    """Run production day by day for every plant.

    arrivals: (runs, n_ships) arrival day per shipment. extra: {(plant id, component id): [(day, qty)]} from a recovery plan.
    Each day a plant builds what demand asks for, limited by its scarcest component; shared components are rationed
    in proportion. Returns per-plant lost production, first stock-out day and (optionally) the inventory trace.
    ponytail: lost production is lost, not caught up later; finished-goods buffer absorbs the first losses.
    """
    by_plant = defaultdict(list)
    for k, s in enumerate(twin.ships):
        by_plant[s["plant"]].append(k)
    out = {}
    for pid, pl in twin.plants.items():
        idx = by_plant[pid]
        rec = _receipts(pl, [twin.ships[k] for k in idx], arrivals[:, idx], runs)
        ci = {c: j for j, c in enumerate(pl.comp_ids)}
        for (p, c), adds in (extra or {}).items():
            if p == pid:
                for day, qty in adds:
                    if 0 <= day < HORIZON:
                        rec[:, ci[c], int(day)] += qty
        inv = np.tile(pl.inv0, (runs, 1))
        want = pl.demand @ pl.q
        uses = pl.q > 0
        lost = np.zeros((runs, len(pl.fg_ids)))
        short_days = np.zeros((runs, len(pl.comp_ids)))
        first = np.full(runs, np.inf)
        cover = np.zeros((HORIZON, len(pl.comp_ids))) if trace else None
        for t in range(HORIZON):
            inv += rec[:, :, t]
            ratio = np.minimum(1.0, inv / np.maximum(want, 1e-9))
            scarce = np.flatnonzero((ratio < 1).any(axis=0))
            if len(scarce):                                    # only components actually short today can limit anything
                factor = np.where(uses[None, :, scarce], ratio[:, None, scarce], 1.0).min(axis=2)
                first = np.where((factor < 1).any(axis=1) & np.isinf(first), t, first)
                short_days[:, scarce] += ratio[:, scarce] < 1
            else:
                factor = np.ones((runs, len(pl.fg_ids)))
            made = pl.demand * factor
            inv = np.maximum(0, inv - made @ pl.q)
            lost += pl.demand - made
            if trace:
                cover[t] = inv[0] / np.maximum(want, 1e-9)
        lost_sales = np.maximum(0, lost - pl.buffer)
        out[pid] = {"lost_units": lost, "lost_sales": lost_sales, "lost_revenue": lost_sales @ pl.price, "lost_margin": lost_sales @ (pl.price - pl.cost),
                    "first_stockout": first, "short_days": short_days, "cover": cover, "demand_revenue": float(pl.demand @ pl.price) * HORIZON}
    return out


def baseline_arrivals(twin):
    return np.array([[s["eta"] for s in twin.ships]])


def totals(sim):
    rev = sum(v["lost_revenue"] for v in sim.values())
    demand = sum(v["demand_revenue"] for v in sim.values())
    return {"lost_revenue": rev, "lost_margin": sum(v["lost_margin"] for v in sim.values()), "service_level": 1 - rev / demand,
            "stockout_component_days": sum(v["short_days"].sum(axis=1) for v in sim.values())}


def monte_carlo(twin, port_id, severity, duration, plan=None, runs=1000, seed=0):
    """`runs` futures: uncertain cut depth and length, and noisy transit times for every inbound shipment."""
    rng = np.random.default_rng(seed)
    sev = np.clip(rng.normal(severity, 0.07, runs), 0.05, 0.95)[:, None]
    dur = (duration * np.exp(rng.normal(0, 0.3, runs)))[:, None]
    etd = np.array([s["etd"] for s in twin.ships])[None, :]
    eta = np.array([s["eta"] for s in twin.ships])[None, :]
    hit = np.array([s["origin"] == port_id and s["etd"] >= 0 for s in twin.ships])[None, :]
    arrive = eta + rng.normal(0, 1.5, (runs, eta.shape[1])) + np.where(hit, port_wait(etd, sev, dur), 0.0)
    if plan:
        for k in plan["expedite"]:                             # flown shipments skip the port queue
            arrive[:, k] = max(twin.ships[k]["etd"], 0) + AIR_DAYS + rng.normal(0, 0.5, runs)
    return totals(simulate(twin, arrive, plan and plan["extra"], runs)), sev[:, 0], dur[:, 0]


def expected_arrivals(twin, hits, plan=None):
    a = baseline_arrivals(twin).copy()
    for k, d in hits:
        a[0, k] += d
    for k in (plan or {}).get("expedite", []):
        a[0, k] = max(twin.ships[k]["etd"], 0) + AIR_DAYS
    return a


def naive_plan(twin, hits):
    """What a planner under pressure does: fly the most delayed shipments first until the air budget is gone."""
    cap = AIR_VALUE_CAP * sum(twin.ships[k]["value"] for k, _ in hits)
    chosen, spent = [], 0.0
    for k, _ in sorted(hits, key=lambda h: -h[1]):
        if spent + twin.ships[k]["value"] <= cap:
            chosen.append(k)
            spent += twin.ships[k]["value"]
    return {"expedite": chosen, "extra": {}, "actions": [], "cost": plan_cost(twin, chosen, [])}


def plan_cost(twin, expedite, actions):
    air = sum(AIR_FIXED + AIR_RATE * twin.ships[k]["value"] for k in expedite)
    return {"airfreight": air, "alternate_supply": sum(a["cost"] for a in actions if a["type"] == "alternate_supplier"),
            "transfers": sum(a["cost"] for a in actions if a["type"] == "warehouse_transfer")}


def optimize(twin, port_id, hits, time_limit=60):
    """Mixed-integer plan: which delayed shipments to fly, what to buy from an unaffected second source and what to pull
    from warehouse reserve, minimising lost margin + airfreight + purchase premium + transfer cost under capacity limits.

    Periods are BUCKET days long. The plan is then judged by the daily simulator, not by this model's own objective.
    """
    periods = HORIZON // BUCKET
    delay = dict(hits)
    solver = pywraplp.Solver.CreateSolver("HIGHS")
    solver.SetTimeLimit(time_limit * 1000)
    inf = solver.infinity()
    obj, x, actions_spec = [], {}, []
    for k in delay:
        x[k] = solver.BoolVar(f"fly{k}")
        obj.append((AIR_FIXED + AIR_RATE * twin.ships[k]["value"]) * x[k])
    solver.Add(sum(twin.ships[k]["value"] * x[k] for k in delay) <= AIR_VALUE_CAP * sum(twin.ships[k]["value"] for k in delay))
    by_plant = defaultdict(list)
    for k, s in enumerate(twin.ships):
        by_plant[s["plant"]].append(k)
    for pid in {twin.ships[k]["plant"] for k in delay}:
        pl = twin.plants[pid]
        ci = {c: j for j, c in enumerate(pl.comp_ids)}
        hit_comps = sorted({ci[twin.ships[k]["part"]] for k in delay if twin.ships[k]["plant"] == pid})
        fgs = np.flatnonzero((pl.q[:, hit_comps] > 0).any(axis=1))            # only products that use a delayed component can lose output
        prod = {(f, b): solver.NumVar(0, pl.demand[f] * BUCKET, "") for f in fgs for b in range(periods)}
        fixed = np.zeros((len(pl.comp_ids), periods))                        # receipts nobody can change
        moved = defaultdict(list)                                            # (component, period) -> [(coefficient, variable)]
        for k in by_plant[pid]:
            s = twin.ships[k]
            j = ci[s["part"]]
            if j not in hit_comps:
                continue
            late = math.ceil(max(0, s["eta"] + delay.get(k, 0)) / BUCKET)      # usable from the first period that starts on or after arrival
            if k in delay:
                fast = math.ceil((max(s["etd"], 0) + AIR_DAYS) / BUCKET)
                if 0 <= late < periods:
                    moved[(j, late)].append((s["qty"], 1 - x[k]))
                if 0 <= fast < periods:
                    moved[(j, fast)].append((s["qty"], x[k]))
            elif 0 <= late < periods:
                fixed[j, late] += s["qty"]
        for j in hit_comps:
            c = pl.comp_ids[j]
            buys = {}
            for sup_id, name, spare, lead, origin in twin.alt.get((pid, c), []):
                if origin == port_id or spare <= 0:                              # the second source must not sit behind the same port
                    continue
                for b in range(int(math.ceil(lead / BUCKET)), periods):
                    v = solver.NumVar(0, spare * BUCKET / 7, "")
                    buys[(sup_id, name, b)] = v
                    obj.append(ALT_PREMIUM * pl.unit_cost[j] * v)
            pulls = {b: solver.NumVar(0, pl.reserve[j], "") for b in range(1, periods)} if pl.reserve[j] > 0 else {}
            if pulls:
                solver.Add(sum(pulls.values()) <= pl.reserve[j])
                obj += [TRANSFER_COST * pl.unit_cost[j] * v for v in pulls.values()]
            inv_prev = pl.inv0[j]
            other = float(pl.demand @ pl.q[:, j]) - float(pl.demand[fgs] @ pl.q[fgs, j])       # use by products outside the model: always served
            for b in range(periods):
                inv = solver.NumVar(0, inf, "")
                inflow = fixed[j, b] + sum(q * e for q, e in moved.get((j, b), [])) + sum(v for (_, _, bb), v in buys.items() if bb == b) + (pulls.get(b, 0))
                solver.Add(inv == inv_prev + inflow - sum(pl.q[f, j] * prod[(f, b)] for f in fgs if pl.q[f, j] > 0) - other * BUCKET)
                inv_prev = inv
            actions_spec.append((pid, j, buys, pulls))
        for f in fgs:
            lost_sales = solver.NumVar(0, inf, "")
            solver.Add(lost_sales >= pl.demand[f] * BUCKET * periods - sum(prod[(f, b)] for b in range(periods)) - pl.buffer[f])
            obj.append((pl.price[f] - pl.cost[f]) * lost_sales)
    solver.Minimize(sum(obj))
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"recovery model has no solution (status {status})")
    expedite = [k for k in delay if x[k].solution_value() > 0.5]
    actions, extra = [], defaultdict(list)
    for pid, j, buys, pulls in actions_spec:
        pl = twin.plants[pid]
        c = pl.comp_ids[j]
        per_sup = defaultdict(list)
        for (sup_id, name, b), v in buys.items():
            if v.solution_value() > 1e-6:
                per_sup[(sup_id, name)].append((b * BUCKET, v.solution_value()))
        for (sup_id, name), adds in per_sup.items():
            qty = sum(q for _, q in adds)
            actions.append({"type": "alternate_supplier", "plant": pl.name, "sku": pl.comp_sku[j], "supplier": name, "qty": round(qty, 1), "first_day": adds[0][0],
                            "cost": round(ALT_PREMIUM * pl.unit_cost[j] * qty, 2)})
            extra[(pid, c)] += adds
        adds = [(b * BUCKET, v.solution_value()) for b, v in pulls.items() if v.solution_value() > 1e-6]
        if adds:
            qty = sum(q for _, q in adds)
            actions.append({"type": "warehouse_transfer", "plant": pl.name, "sku": pl.comp_sku[j], "supplier": "regional warehouse reserve", "qty": round(qty, 1),
                            "first_day": adds[0][0], "cost": round(TRANSFER_COST * pl.unit_cost[j] * qty, 2)})
            extra[(pid, c)] += adds
    return {"expedite": expedite, "extra": dict(extra), "actions": actions, "cost": plan_cost(twin, expedite, actions),
            "solver": {"name": "HiGHS via OR-Tools", "status": "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible (time limit)",
                       "variables": solver.NumVariables(), "constraints": solver.NumConstraints(), "wall_ms": solver.wall_time(), "objective": solver.Objective().Value()}}


def from_world(w):
    """Build the twin straight from `world.build()` output (tests and the offline evaluation; the API loads from Postgres)."""
    line = {ln[1]: ln for ln in w["po_lines"]}
    po = {p[0]: p for p in w["purchase_orders"]}
    return build(
        w["now"],
        [{"id": f[0], "facility_type": f[2], "name": f[3]} for f in w["facilities"]],
        [{"id": p[0], "sku": p[2], "unit_cost": p[4], "part_type": p[5], "home_facility_id": p[6], "weekly_demand": p[7], "unit_price": p[8]} for p in w["parts"]],
        [{"parent_part_id": b[0], "child_part_id": b[1], "qty": b[2]} for b in w["bom"]],
        [{"supplier_id": s[0], "part_id": s[1], "lead_time_days": s[2], "capacity_per_week": s[3], "allocation_pct": s[4]} for s in w["supplier_parts"]],
        [{"id": s[0], "name": s[2], "origin_facility_id": s[6]} for s in w["suppliers"]],
        [{"id": s[0], "destination_id": s[4], "part_id": line[s[1]][2], "qty": line[s[1]][3], "origin_id": s[3], "etd": s[6], "eta": s[7], "supplier_id": po[s[1]][1],
          "mode": s[2], "status": s[8], "carrier": s[5]} for s in w["shipments"] if s[8] != "delivered"],
        [{"facility_id": i[0], "part_id": i[1], "on_hand": i[3]} for i in w["inventory"]])
