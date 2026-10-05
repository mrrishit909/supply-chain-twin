"""Load the synthetic network into Postgres and train the models.

    python -m sc.seed [--if-empty] [--scale 1.0]
"""
import argparse
import hashlib
import uuid

from psycopg.types.json import Jsonb

from core import db

from . import models, world

TENANTS = {"acme": ("Acme Manufacturing", 1.0, 3), "other": ("Second Tenant Ltd", 0.05, 9)}
ROLES = ["planner", "viewer", "admin"]


def token(tenant, role):
    """Demo credentials for local use only."""
    return f"{tenant}-{role}-demo"


def write(c, w, name, key):
    t = w["tenant"]
    c.execute("INSERT INTO tenants (id, name) VALUES (%s,%s)", [t, name])
    for role in ROLES:
        c.execute("INSERT INTO api_tokens VALUES (%s,%s,%s,%s,%s)", [hashlib.sha256(token(key, role).encode()).hexdigest(), t, uuid.uuid5(t, role), f"{role} ({name})", role])
    db.copy(c, "facilities", ["id", "tenant_id", "facility_type", "name", "lat", "lon", "capacity_json"], [(*f[:6], Jsonb(f[6])) for f in w["facilities"]])
    db.copy(c, "suppliers", ["id", "tenant_id", "name", "tier", "country", "risk_score", "origin_facility_id"], w["suppliers"])
    db.copy(c, "parts", ["id", "tenant_id", "sku", "description", "unit_cost", "part_type", "home_facility_id", "weekly_demand", "unit_price"], w["parts"])
    db.copy(c, "bom_edges", ["parent_part_id", "child_part_id", "qty"], w["bom"])
    db.copy(c, "supplier_parts", ["supplier_id", "part_id", "lead_time_days", "capacity_per_week", "allocation_pct"], w["supplier_parts"])
    db.copy(c, "purchase_orders", ["id", "supplier_id", "facility_id", "status", "order_date", "promised_date", "value"], w["purchase_orders"])
    db.copy(c, "po_lines", ["id", "po_id", "part_id", "qty", "unit_price"], w["po_lines"])
    db.copy(c, "shipments", ["id", "po_id", "mode", "origin_id", "destination_id", "carrier", "etd", "eta", "status", "delivered_at"], w["shipments"])
    db.copy(c, "inventory_positions", ["facility_id", "part_id", "observed_at", "on_hand", "allocated", "in_transit"], w["inventory"])
    db.copy(c, "port_congestion", ["facility_id", "week_start", "congestion_index"], w["congestion"])


def main(if_empty=False, scale=None):
    db.migrate()
    with db.tx() as c:
        if if_empty and c.execute("SELECT 1 FROM tenants LIMIT 1").fetchone():
            return print("already seeded")
    out = {}
    for key, (name, default_scale, seed) in TENANTS.items():
        w = world.build(scale if (scale and key == "acme") else default_scale, seed, tenant_key=key)
        with db.tx() as c:
            write(c, w, name, key)
            if key == "acme":      # models are trained on the main tenant's two years of deliveries and registered with their held-out scores
                eta, eta_m = models.train_eta(w["history"])
                risk, risk_m, score, rate = models.train_risk(w["history"])
                snapshot = hashlib.sha256(repr(w["history"][:2000]).encode()).hexdigest()[:16]
                for nm, ver, art, m in [("eta-forecast", models.ETA_VERSION, eta, eta_m), ("supplier-delay", models.RISK_VERSION, {"clf": risk, "rate": rate}, risk_m)]:
                    c.execute("INSERT INTO model_registry (name, version, data_snapshot, metrics, baseline, artifact) VALUES (%s,%s,%s,%s,%s,%s)",
                              [nm, ver, snapshot, Jsonb({**m["model"], "n_train": m["n_train"], "n_test": m["n_test"]}), Jsonb(m["baseline"]), models.dumps(art)])
                db.many(c, "UPDATE suppliers SET risk_score = %s WHERE id = %s", [(score.get(i), s["id"]) for i, s in enumerate(w["sup_meta"])])
                out["metrics"] = {"eta": eta_m, "risk": risk_m}
        out[key] = w
        print(f"{name}: {len(w['suppliers'])} suppliers, {len(w['facilities'])} facilities, {len(w['parts'])} parts, {len(w['shipments'])} shipments")
    print("demo tokens:", ", ".join(token(t, r) for t in TENANTS for r in ROLES))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-empty", action="store_true")
    ap.add_argument("--scale", type=float)
    a = ap.parse_args()
    main(a.if_empty, a.scale)
