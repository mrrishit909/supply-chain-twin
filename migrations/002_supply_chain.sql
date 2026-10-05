-- Domain model. Names follow the blueprint's persistence model; additions are marked (+).
-- The blueprint types locations as PostGIS geography. No spatial query runs in SQL in this build (the map draws
-- points, disruptions attach to a facility), so locations are plain lat/lon columns and PostGIS is not required.

CREATE TABLE facilities (
  id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, facility_type text NOT NULL CHECK (facility_type IN ('port','plant','warehouse')),
  name text NOT NULL, lat double precision NOT NULL, lon double precision NOT NULL, capacity_json jsonb NOT NULL, UNIQUE (tenant_id, name));
SELECT enable_tenant_rls('facilities');

CREATE TABLE suppliers (
  id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, tier int NOT NULL, country text NOT NULL, risk_score numeric,
  origin_facility_id uuid REFERENCES facilities);                                                                   -- (+) the port it ships through, if any
SELECT enable_tenant_rls('suppliers');

CREATE TABLE parts (
  id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, sku text NOT NULL, description text, unit_cost numeric,
  part_type text NOT NULL, home_facility_id uuid REFERENCES facilities, weekly_demand numeric, unit_price numeric,   -- (+)
  UNIQUE (tenant_id, sku));
SELECT enable_tenant_rls('parts');

CREATE TABLE bom_edges (parent_part_id uuid REFERENCES parts, child_part_id uuid REFERENCES parts, qty numeric NOT NULL, PRIMARY KEY (parent_part_id, child_part_id));
CREATE TABLE supplier_parts (
  supplier_id uuid REFERENCES suppliers, part_id uuid REFERENCES parts, lead_time_days numeric NOT NULL, capacity_per_week numeric NOT NULL, allocation_pct numeric NOT NULL,
  PRIMARY KEY (supplier_id, part_id));
CREATE INDEX supplier_parts_part ON supplier_parts (part_id);

CREATE TABLE purchase_orders (
  id uuid PRIMARY KEY, supplier_id uuid NOT NULL REFERENCES suppliers, facility_id uuid NOT NULL REFERENCES facilities, status text NOT NULL,
  order_date date NOT NULL, promised_date date NOT NULL, value numeric NOT NULL);
CREATE INDEX purchase_orders_supplier ON purchase_orders (supplier_id);
CREATE TABLE po_lines (id uuid PRIMARY KEY, po_id uuid NOT NULL REFERENCES purchase_orders, part_id uuid NOT NULL REFERENCES parts, qty numeric NOT NULL, unit_price numeric NOT NULL);
CREATE INDEX po_lines_po ON po_lines (po_id);

CREATE TABLE shipments (
  id uuid PRIMARY KEY, po_id uuid NOT NULL REFERENCES purchase_orders, mode text NOT NULL, origin_id uuid REFERENCES facilities, destination_id uuid NOT NULL REFERENCES facilities,
  carrier text NOT NULL, etd timestamptz NOT NULL, eta timestamptz NOT NULL, status text NOT NULL CHECK (status IN ('planned','in_transit','delivered')),
  delivered_at timestamptz);                                                                                        -- (+)
CREATE INDEX shipments_open ON shipments (status) WHERE status <> 'delivered';
CREATE INDEX shipments_po ON shipments (po_id);

CREATE TABLE inventory_positions (
  facility_id uuid REFERENCES facilities, part_id uuid REFERENCES parts, observed_at timestamptz, on_hand numeric NOT NULL, allocated numeric NOT NULL, in_transit numeric NOT NULL,
  PRIMARY KEY (facility_id, part_id, observed_at));

CREATE TABLE port_congestion (facility_id uuid REFERENCES facilities, week_start date, congestion_index numeric NOT NULL, PRIMARY KEY (facility_id, week_start));   -- (+)

CREATE TABLE disruption_events (
  id uuid PRIMARY KEY, event_type text NOT NULL, severity numeric NOT NULL, lat double precision, lon double precision, starts_at timestamptz NOT NULL, ends_at timestamptz NOT NULL,
  source text NOT NULL, payload jsonb NOT NULL,
  tenant_id uuid NOT NULL REFERENCES tenants);                                                                      -- (+) injected what-if events belong to a tenant
SELECT enable_tenant_rls('disruption_events');

CREATE TABLE scenarios (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, assumptions jsonb NOT NULL, created_by uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now());                                                                   -- (+)
SELECT enable_tenant_rls('scenarios');
CREATE TABLE scenario_results (id uuid PRIMARY KEY, scenario_id uuid NOT NULL REFERENCES scenarios ON DELETE CASCADE, metric text NOT NULL, entity_ref text NOT NULL,
  value numeric NOT NULL, detail jsonb NOT NULL);
CREATE INDEX scenario_results_scenario ON scenario_results (scenario_id, metric);
