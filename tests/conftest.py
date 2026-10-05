import os

import psycopg
import pytest

ADMIN = os.environ.get("DATABASE_URL", "postgresql://app:app-dev-only@127.0.0.1:55103/app")
os.environ["DATABASE_URL"] = ADMIN.rsplit("/", 1)[0] + "/app_test"


@pytest.fixture(scope="session")
def world():
    """A fresh database seeded with a one-fifth-size network for the main tenant and a tiny one for the second."""
    with psycopg.connect(ADMIN, autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS app_test WITH (FORCE)")
        c.execute("CREATE DATABASE app_test")
    from core import db
    from sc import seed
    out = seed.main(scale=0.2)
    yield out
    db.close()


@pytest.fixture(scope="session")
def client(world):
    from fastapi.testclient import TestClient

    from sc.api import app
    return TestClient(app)


def bearer(tenant, role):
    return {"Authorization": f"Bearer {tenant}-{role}-demo"}
