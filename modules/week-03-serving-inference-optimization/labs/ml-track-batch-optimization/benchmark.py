"""
Benchmarks batched vs. sequential scoring to prove batching's throughput win.

Run with `make benchmark` or `python benchmark.py`. Uses an in-process TestClient
so no server needs to be running. The feature-store failure rate is forced to
0 for this benchmark so the measurement isolates batching's effect from the
separate retry-simulation lesson (see app/adapters/feature_store.py) — with
the default 20% failure rate, most of the wall-clock time is retry sleeps,
not real work, and it would also risk counting a hard-failed batch response as
a valid (and misleadingly fast) timing. Absolute numbers vary by machine, but
the relative speedup direction is the point.
"""
import os
import time

os.environ["FEATURE_STORE_FAILURE_RATE"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

N_TRANSACTIONS = 100
N_TRIALS = 30


def _transaction(i: int) -> dict:
    return {
        "transaction_id": f"txn-{i}",
        "amount": 10.0 + i,
        "merchant_category": "electronics",
    }


def _time_sequential(client: TestClient) -> float:
    start = time.perf_counter()
    for i in range(N_TRANSACTIONS):
        response = client.post("/score", json=_transaction(i))
        assert response.status_code == 200, (
            f"sequential /score call failed unexpectedly: {response.status_code} {response.text}"
        )
    return time.perf_counter() - start


def _time_batched(client: TestClient) -> float:
    payload = {"transactions": [_transaction(i) for i in range(N_TRANSACTIONS)]}
    start = time.perf_counter()
    response = client.post("/score/batch", json=payload)
    elapsed = time.perf_counter() - start
    assert response.status_code == 200, (
        f"batched /score/batch call failed unexpectedly: {response.status_code} {response.text}"
    )
    return elapsed


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    index = min(int(len(values) * pct), len(values) - 1)
    return values[index]


def main() -> None:
    with TestClient(app) as client:
        sequential_times = [_time_sequential(client) for _ in range(N_TRIALS)]
        batched_times = [_time_batched(client) for _ in range(N_TRIALS)]

    seq_p50, seq_p95 = _percentile(sequential_times, 0.5), _percentile(
        sequential_times, 0.95
    )
    batch_p50, batch_p95 = _percentile(batched_times, 0.5), _percentile(
        batched_times, 0.95
    )

    report = (
        f"Benchmark: {N_TRANSACTIONS} transactions x {N_TRIALS} trials\n"
        f"(feature_store_failure_rate forced to 0 for this measurement)\n"
        f"{'':20}{'p50 (s)':>12}{'p95 (s)':>12}\n"
        f"{'sequential':20}{seq_p50:>12.4f}{seq_p95:>12.4f}\n"
        f"{'batched':20}{batch_p50:>12.4f}{batch_p95:>12.4f}\n"
        f"speedup (p50): {seq_p50 / batch_p50:.2f}x\n"
    )
    print(report)
    with open("benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
