# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) with the full synthetic network: 500 suppliers, 5,000 parts,
294,897 shipments (38,696 open or planned). Reproduce with `make demo && make load-test`.

## Map state (`GET /v1/map/state`), closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 1417 | 141.7 req/s | 6.6 ms | 7.7 ms | 9.4 ms | 0.00% |
| 8 | 1588 | 158.1 req/s | 48.1 ms | 65.4 ms | 76.2 ms | 0.00% |
| 32 | 1276 | 125.5 req/s | 201.4 ms | 250.9 ms | 279.1 ms | 0.00% |

Target: p95 under 750 ms for interactive reads: met. The first version rebuilt the payload from 38,696 in-memory shipments on every
request (37 ms each, p95 1.3 s at 32 clients). The network only changes when the twin is rebuilt, so the payload is now built once per
process and tenant.

## Compute, from the job results and `docs/evaluation.md`

| Work | Time |
|---|---|
| Seed: generate, COPY 885k rows, train two models | about 25 s |
| Build the twin from Postgres (first request per process) | about 2 s |
| Expected-case exposure (two 84-day simulations of 10 plants) | under 0.5 s |
| Monte Carlo, 1,000 futures x 84 days x 10 plants (run twice: with and without the cut) | about 8 s |
| Recovery MIP, 78,437 variables (HiGHS) | 3 to 5 s; 16 s for a 42-day cut |

Long work runs as jobs on the worker (`202` + `GET /v1/jobs/{id}`), so the API stays responsive. Not measured: concurrent scenario jobs
(one worker processes them one at a time) and networks larger than this one.
