"""Deterministic synthetic supply network: suppliers, ports, plants, warehouses, a multi-level BOM and two years of orders.

`build()` returns plain row tuples in table order, so the same data seeds Postgres, trains the models and feeds the tests.
Port and city coordinates are real places; every company, part and order is invented. Day 0 is the day the seed runs.
"""
import datetime
import math
import uuid

import numpy as np

HORIZON = 84          # days the twin plans ahead
HISTORY = 730
# name, country, lat, lon, region
PORTS = [("Shanghai", "CN", 31.23, 121.47, "asia"), ("Shenzhen", "CN", 22.54, 114.06, "asia"), ("Singapore", "SG", 1.26, 103.82, "asia"),
         ("Busan", "KR", 35.10, 129.04, "asia"), ("Nhava Sheva", "IN", 18.95, 72.95, "asia"), ("Rotterdam", "NL", 51.95, 4.14, "europe"),
         ("Hamburg", "DE", 53.54, 9.98, "europe"), ("Los Angeles", "US", 33.74, -118.26, "americas"), ("Savannah", "US", 32.08, -81.09, "americas"),
         ("Houston", "US", 29.73, -95.27, "americas"), ("Santos", "BR", -23.96, -46.33, "americas"), ("Veracruz", "MX", 19.20, -96.13, "americas")]
# name, country, lat, lon, gateway port
PLANTS = [("Detroit Plant", "US", 42.33, -83.05, "Savannah"), ("Greenville Plant", "US", 34.85, -82.40, "Savannah"), ("Reno Plant", "US", 39.53, -119.81, "Los Angeles"),
          ("Phoenix Plant", "US", 33.45, -112.07, "Los Angeles"), ("San Antonio Plant", "US", 29.42, -98.49, "Houston"), ("Monterrey Plant", "MX", 25.69, -100.32, "Veracruz"),
          ("Guadalajara Plant", "MX", 20.66, -103.35, "Veracruz"), ("Stuttgart Plant", "DE", 48.78, 9.18, "Hamburg"), ("Wroclaw Plant", "PL", 51.11, 17.04, "Hamburg"),
          ("Eindhoven Plant", "NL", 51.44, 5.47, "Rotterdam")]
WAREHOUSES = [("Chicago", 41.88, -87.63), ("Dallas", 32.78, -96.80), ("Atlanta", 33.75, -84.39), ("Columbus", 39.96, -83.00), ("Memphis", 35.15, -90.05), ("Denver", 39.74, -104.99),
              ("Seattle", 47.61, -122.33), ("Newark", 40.74, -74.17), ("Miami", 25.76, -80.19), ("Kansas City", 39.10, -94.58), ("Toronto", 43.65, -79.38), ("Mexico City", 19.43, -99.13),
              ("Queretaro", 20.59, -100.39), ("Tijuana", 32.51, -117.04), ("Lyon", 45.76, 4.84), ("Milan", 45.46, 9.19), ("Madrid", 40.42, -3.70), ("Prague", 50.08, 14.44),
              ("Vienna", 48.21, 16.37), ("Antwerp", 51.22, 4.40), ("Manchester", 53.48, -2.24), ("Gothenburg", 57.71, 11.97), ("Budapest", 47.50, 19.04), ("Munich", 48.14, 11.58),
              ("Leipzig", 51.34, 12.37), ("Lille", 50.63, 3.06), ("Bilbao", 43.26, -2.93), ("Katowice", 50.26, 19.02)]
# supplier country -> (share, candidate origin ports; [] = overland to plants on the same continent)
COUNTRIES = {"CN": (0.34, ["Shanghai", "Shanghai", "Shanghai", "Shenzhen", "Shenzhen"]), "US": (0.16, []), "MX": (0.08, []), "DE": (0.08, ["Hamburg"]), "KR": (0.06, ["Busan"]),
             "IN": (0.06, ["Nhava Sheva"]), "VN": (0.07, ["Singapore"]), "TH": (0.05, ["Singapore"]), "PL": (0.04, ["Hamburg"]), "BR": (0.03, ["Santos"]), "NL": (0.03, ["Rotterdam"])}
CONTINENT = {"US": "americas", "MX": "americas", "BR": "americas", "DE": "europe", "PL": "europe", "NL": "europe"}
CARRIERS = ["Borealis Line", "Tidewater Shipping", "Meridian Ocean", "Kestrel Freight", "Halcyon Logistics"]
CARRIER_DAYS = [-1.0, 0.0, 0.5, 1.5, 3.0]


def km(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    return 6371 * math.acos(min(1, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)))


def build(scale=1.0, seed=3, today=None, tenant_key="acme"):
    rng = np.random.default_rng(seed)
    today = today or datetime.datetime.now(datetime.UTC).date()
    now = datetime.datetime.combine(today, datetime.time(), datetime.UTC)
    day = lambda d: now + datetime.timedelta(days=float(d))                          # noqa: E731
    uid = lambda: uuid.UUID(bytes=rng.bytes(16), version=4)                          # noqa: E731
    tenant = uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_key}.supply-chain-twin.example")
    w = {"tenant": tenant, "now": now, "today": today}

    # ---- facilities -------------------------------------------------------------------------------------
    fac, port_id, plant_id = [], {}, {}
    for name, country, lat, lon, region in PORTS:
        port_id[name] = uid()
        fac.append((port_id[name], tenant, "port", name, lat, lon, {"country": country, "region": region, "utilization": 0.8}))
    for name, country, lat, lon, gateway in PLANTS:
        plant_id[name] = uid()
        fac.append((plant_id[name], tenant, "plant", name, lat, lon, {"country": country, "gateway_port": gateway}))
    for name, lat, lon in WAREHOUSES:
        fac.append((uid(), tenant, "warehouse", f"{name} DC", lat, lon, {}))
    w["facilities"] = fac
    plants = [p[0] for p in PLANTS]

    # ---- suppliers --------------------------------------------------------------------------------------
    n_sup = max(40, int(500 * scale))
    countries = rng.choice(list(COUNTRIES), n_sup, p=np.array([v[0] for v in COUNTRIES.values()]))
    tiers = rng.choice([1, 2, 3], n_sup, p=[0.3, 0.4, 0.3])
    sup = [{"id": uid(), "name": f"Supplier {i:03d} ({c})", "tier": int(t), "country": c, "port": (rng.choice(COUNTRIES[c][1]) if COUNTRIES[c][1] else None),
            "reliability": float(rng.beta(2, 6) * 6)} for i, (c, t) in enumerate(zip(countries, tiers))]            # extra days late, supplier's own doing

    # ---- parts and the multi-level BOM ------------------------------------------------------------------
    n_fg, n_sub, n_comp = max(20, int(200 * scale)), max(40, int(800 * scale)), max(200, int(4000 * scale))
    comp_cost = np.round(np.exp(rng.normal(1.6, 1.2, n_comp)).clip(0.2, 600), 2)
    comp_plant = np.arange(n_comp) % 10
    fg_plant, sub_plant = np.arange(n_fg) % 10, np.arange(n_sub) % 10
    parts, bom, comp_ids, sub_ids, fg_ids = [], [], [uid() for _ in range(n_comp)], [uid() for _ in range(n_sub)], [uid() for _ in range(n_fg)]
    comp_of_plant = [np.flatnonzero(comp_plant == p) for p in range(10)]
    sub_of_plant = [np.flatnonzero(sub_plant == p) for p in range(10)]
    sub_children = {}
    for s in range(n_sub):                               # sub-assembly -> 3-6 components of its plant (round-robin so every component is used)
        pool = comp_of_plant[sub_plant[s]]
        k = int(rng.integers(3, 7))
        start = (s // 10) * 5
        kids = {int(pool[(start + j) % len(pool)]) for j in range(k)}
        sub_children[s] = [(c, int(rng.integers(1, 5))) for c in sorted(kids)]
        bom += [(sub_ids[s], comp_ids[c], q) for c, q in sub_children[s]]
    q_flat = np.zeros((n_fg, n_comp))                    # exploded requirement: component units per finished unit
    for f in range(n_fg):
        pool = sub_of_plant[fg_plant[f]]                  # four sub-assemblies each, dealt round-robin so every one is used
        subs = [pool[((f // 10) * 4 + j) % len(pool)] for j in range(4)]
        for s in subs:
            q = int(rng.integers(1, 3))
            bom.append((fg_ids[f], sub_ids[s], q))
            for c, qc in sub_children[int(s)]:
                q_flat[f, c] += q * qc
        for c in rng.choice(comp_of_plant[fg_plant[f]], int(rng.integers(3, 7)), replace=False):
            q = int(rng.integers(1, 4))
            bom.append((fg_ids[f], comp_ids[int(c)], q))
            q_flat[f, c] += q
    bom = list({(a, b): (a, b, q) for a, b, q in bom}.values())
    fg_cost = q_flat @ comp_cost
    fg_demand = np.round(np.exp(rng.normal(5.0, 0.8, n_fg)).clip(20, 3000))                  # units per week
    fg_price = np.round(fg_cost * rng.uniform(1.6, 2.4, n_fg), 2)
    for c in range(n_comp):
        parts.append((comp_ids[c], tenant, f"CMP-{c:05d}", f"Component {c:05d}", float(comp_cost[c]), "component", plant_id[plants[comp_plant[c]]], None, None))
    for s in range(n_sub):
        parts.append((sub_ids[s], tenant, f"SUB-{s:04d}", f"Sub-assembly {s:04d}", None, "subassembly", plant_id[plants[sub_plant[s]]], None, None))
    for f in range(n_fg):
        parts.append((fg_ids[f], tenant, f"FG-{f:04d}", f"Finished product {f:04d}", float(round(fg_cost[f], 2)), "finished", plant_id[plants[fg_plant[f]]], float(fg_demand[f]), float(fg_price[f])))
    w["parts"], w["bom"] = parts, bom
    usage_day = (q_flat.T @ fg_demand) / 7                                                   # component units per day

    # ---- sourcing: 1-3 suppliers per component, 30% single-sourced ---------------------------------------
    streams, supplier_parts = [], []
    wait_share = np.zeros(n_comp)          # sum over a component's streams of allocation x days until that stream's next arrival
    usable = {}
    for p, (name, country, *_rest) in enumerate(PLANTS):                                     # overland suppliers only serve their own continent
        usable[p] = [i for i, s in enumerate(sup) if s["port"] or CONTINENT.get(s["country"]) == CONTINENT[country]]
    for c in range(n_comp):
        if usage_day[c] <= 0:
            continue
        p = int(comp_plant[c])
        k = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
        chosen = rng.choice(usable[p], k, replace=False)
        alloc = {1: [1.0], 2: [0.7, 0.3], 3: [0.5, 0.3, 0.2]}[k]
        for s, a in zip(chosen, alloc):
            s = int(s)
            plant = PLANTS[p]
            if sup[s]["port"]:
                o = next(x for x in PORTS if x[0] == sup[s]["port"])
                g = next(x for x in PORTS if x[0] == plant[4])
                base = km(o[2], o[3], g[2], g[3]) * 1.25 / 710 + 4 + 3                        # sea leg + port handling + inland
            else:
                base = 2 + rng.uniform(1, 5)
            cadence = int(rng.choice([2, 3, 4]))
            weekly = usage_day[c] * 7 * a
            streams.append({"comp": c, "sup": s, "plant": p, "alloc": a, "cadence": cadence, "offset": int(rng.integers(0, cadence * 7)), "base": float(base),
                            "carrier": int(rng.integers(0, 5)), "weekly": float(weekly)})
            wait_share[c] += a * ((streams[-1]["offset"] - HISTORY - 1) % (cadence * 7) + 1)      # days to the first arrival after day 0
            supplier_parts.append((sup[s]["id"], comp_ids[c], round(base + 2), float(round(weekly * rng.uniform(1.3, 2.5), 1)), a * 100))
    w["supplier_parts"] = supplier_parts
    # supplier risk score shown in the UI is filled in by the model after training; start neutral
    w["suppliers"] = [(s["id"], tenant, s["name"], s["tier"], s["country"], None, port_id.get(s["port"])) for s in sup]

    # ---- port congestion index, weekly, AR(1) -----------------------------------------------------------
    weeks = (HISTORY + HORIZON) // 7 + 2
    cong = np.zeros((len(PORTS), weeks))
    for t in range(1, weeks):
        cong[:, t] = (0.8 * cong[:, t - 1] + rng.normal(0, 0.25, len(PORTS))).clip(0, 2)
    w["congestion"] = [(port_id[PORTS[i][0]], (today - datetime.timedelta(days=HISTORY - 7 * t)), float(round(cong[i, t], 3))) for i in range(len(PORTS)) for t in range(weeks)]
    port_index = {p[0]: i for i, p in enumerate(PORTS)}

    # ---- orders: one PO, one line, one shipment per replenishment ----------------------------------------
    pos, lines, ships, hist = [], [], [], []
    for st in streams:
        s, c, step = sup[st["sup"]], st["comp"], st["cadence"] * 7
        promised = round(st["base"] + 2)
        arrivals = np.arange(-HISTORY + st["offset"], HORIZON + promised + step, step)
        etd = arrivals - promised
        etd = etd[etd <= HORIZON]
        month = np.array([(today + datetime.timedelta(days=int(e))).month for e in etd])
        pi = port_index.get(s["port"], -1)
        season = np.where(np.isin(month, [9, 10, 11]), 2.0, 0.0) + (np.where(month == 2, 3.0, 0.0) if s["country"] == "CN" else 0.0)
        cg = cong[pi, ((etd + HISTORY) // 7).clip(0, weeks - 1)] if pi >= 0 else np.zeros(len(etd))
        weather = np.where(rng.random(len(etd)) < 0.06, rng.exponential(4, len(etd)), 0.0)
        actual = (st["base"] + (CARRIER_DAYS[st["carrier"]] + season + 3.0 * cg if pi >= 0 else 0) + weather + s["reliability"] * rng.random(len(etd))) * np.exp(rng.normal(0, 0.06, len(etd)))
        qty = float(round(st["weekly"] * st["cadence"], 1))
        origin = port_id[s["port"]] if pi >= 0 else None
        dest, mode = plant_id[plants[st["plant"]]], "ocean" if pi >= 0 else "truck"
        for e, a, m, g, wx in zip(etd, actual, month, cg, weather):
            e = int(e)
            delivered = e + a
            status = "delivered" if (delivered <= 0 and e + promised <= 0) else "in_transit" if e <= 0 else "planned"      # stock on hand assumes arrival on the promised day
            po, sh = uid(), uid()
            pos.append((po, s["id"], dest, "closed" if status == "delivered" else "open", today + datetime.timedelta(days=e - 5), today + datetime.timedelta(days=e + promised),
                        float(round(qty * comp_cost[c], 2))))
            lines.append((uid(), po, comp_ids[c], qty, float(comp_cost[c])))
            ships.append((sh, po, mode, origin, dest, CARRIERS[st["carrier"]], day(e), day(e + promised), status, day(delivered) if status == "delivered" else None))
            if status == "delivered":
                hist.append((e, st["sup"], pi, st["plant"], st["carrier"], int(m), float(g), st["base"], promised, float(a), s["tier"], st["alloc"]))
    w.update(purchase_orders=pos, po_lines=lines, shipments=ships, history=hist, sup_meta=sup, streams=streams)

    # ---- inventory ----------------------------------------------------------------------------------------
    # Plants hold each component's safety stock plus what is needed until the next scheduled arrival, so an undisturbed
    # network never stocks out. A quarter of components also have a reserve at a regional warehouse. Finished goods
    # are held as a few days of buffer.
    safety = rng.uniform(3, 12, n_comp)
    w["inventory"] = [(plant_id[plants[comp_plant[c]]], comp_ids[c], now, float(round(usage_day[c] * (safety[c] + wait_share[c]), 1)), 0.0, 0.0) for c in range(n_comp) if usage_day[c] > 0]
    wh = [f for f in fac if f[2] == "warehouse"]
    for c in np.flatnonzero((rng.random(n_comp) < 0.25) & (usage_day > 0)):
        home = wh[:14] if comp_plant[c] < 7 else wh[14:]                                       # same continent as the plant
        w["inventory"].append((home[int(rng.integers(0, len(home)))][0], comp_ids[c], now, float(round(usage_day[c] * rng.uniform(3, 10), 1)), 0.0, 0.0))
    fg_buffer_days = rng.uniform(3, 10, n_fg)
    w["inventory"] += [(plant_id[plants[fg_plant[f]]], fg_ids[f], now, float(round(fg_demand[f] / 7 * fg_buffer_days[f], 1)), 0.0, 0.0) for f in range(n_fg)]
    return w
