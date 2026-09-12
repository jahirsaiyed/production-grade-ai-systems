"""
Benchmarks batched vs. sequential scoring to prove batching's throughput win.

Run with `make benchmark` or `python benchmark.py`. Uses an in-process TestClient
so no server needs to be running; absolute numbers vary by machine, but the
relative speedup direction is the point.
"""
import time

from fastapi.testclient import TestClient

from app.main import app

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
        client.post("/score", json=_transaction(i))
    return time.perf_counter() - start


def _time_batched(client: TestClient) -> float:
    payload = {"transactions": [_transaction(i) for i in range(N_TRANSACTIONS)]}
    start = time.perf_counter()
    client.post("/score/batch", json=payload)
    return time.perf_counter() - start


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
