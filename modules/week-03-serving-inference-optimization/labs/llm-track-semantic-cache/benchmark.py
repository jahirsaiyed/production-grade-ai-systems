"""
Benchmarks semantic-cache hit vs. miss latency and illustrates cost savings.

Run with `make benchmark` or `python benchmark.py`. Uses an in-process TestClient
so no server needs to be running.
"""
import time

from fastapi.testclient import TestClient

from app.main import app

N_TRIALS = 20
REPEATED_QUESTION = "How many vacation days do I get?"
# Illustrative gpt-4o-mini-class estimate, not a real pricing guarantee — see
# Week 3's Exercise 5 for computing a more realistic figure.
ASSUMED_COST_PER_LLM_CALL_USD = 0.0006


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    index = min(int(len(values) * pct), len(values) - 1)
    return values[index]


def main() -> None:
    with TestClient(app) as client:
        miss_times = []
        hit_times = []
        for i in range(N_TRIALS):
            start = time.perf_counter()
            client.post("/ask", json={"question": f"unique question number {i}"})
            miss_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            client.post("/ask", json={"question": REPEATED_QUESTION})
            hit_times.append(time.perf_counter() - start)

    miss_p50 = _percentile(miss_times, 0.5)
    hit_p50 = _percentile(hit_times, 0.5)
    cost_saved = N_TRIALS * ASSUMED_COST_PER_LLM_CALL_USD

    report = (
        f"Benchmark: {N_TRIALS} cache-miss vs. cache-hit trials\n"
        f"{'':20}{'p50 (s)':>12}\n"
        f"{'cache miss':20}{miss_p50:>12.4f}\n"
        f"{'cache hit':20}{hit_p50:>12.4f}\n"
        f"speedup (p50): {miss_p50 / hit_p50:.2f}x\n"
        f"Illustrative cost saved over {N_TRIALS} avoided LLM calls "
        f"(at ${ASSUMED_COST_PER_LLM_CALL_USD}/call): ${cost_saved:.4f}\n"
    )
    print(report)
    with open("benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
