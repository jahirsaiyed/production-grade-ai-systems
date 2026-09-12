# Production-Grade AI Systems — Self-Study Course

> **Disclaimer:** This is an independent, unofficial study companion inspired
> by the publicly published syllabus of ByteByteGo Live's "Build Production
> Grade AI Systems" course
> (https://live.bytebytego.com/courses/production-ai). It is not affiliated
> with, endorsed by, or produced by ByteByteGo or the course instructor. No
> proprietary lecture content, slides, or recordings are reproduced here —
> only the publicly listed topic outline is used as a curriculum skeleton;
> all explanations, exercises, and code in this repo are written
> independently.

A hands-on, self-study course on shipping AI systems to production: turning
prototypes into layered services, building enterprise RAG pipelines, cutting
inference cost and latency, evaluating and monitoring production AI, and
securing and scaling it.

## Who this is for

Engineers with basic Python (creating a virtual environment, installing
packages with pip) who want a structured, hands-on path from "it works in a
notebook" to "it's a production service." No extensive ML background required.

## Prerequisites

- Python 3.11+
- Git
- Docker (for the containerization labs)

## Time commitment

Roughly 4-7 hours per week, self-paced — work through a module whenever fits
your schedule.

## Course map

| Week | Topic | Status |
|---|---|---|
| 1 | [From Prototype to Production](modules/week-01-prototype-to-production/README.md) | Fully built |
| 2 | [Data and Model Pipelines](modules/week-02-data-and-model-pipelines/README.md) | Fully built |
| 3 | [Serving, Inference, and Optimization](modules/week-03-serving-inference-optimization/README.md) | Fully built |
| 4 | [Evaluation and Monitoring](modules/week-04-evaluation-and-monitoring/README.md) | Fully built |
| 5 | [Security and Governance](modules/week-05-security-and-governance/README.md) | Coming soon |
| 6 | [Scaling](modules/week-06-scaling/README.md) | Coming soon |

The full source syllabus is preserved at [`docs/course-outline.md`](docs/course-outline.md).

## Quickstart (Week 1)

```bash
git clone <this-repo-url>
cd production-grade-ai-systems/modules/week-01-prototype-to-production/labs/ml-track-fraud-detection
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
make setup
make train   # generates the versioned model artifact
make test
make run
```

Repeat inside `labs/llm-track-qa-service` for the LLM track — it works with
zero API key (mock mode) out of the box.

## License

MIT — see [`LICENSE`](LICENSE).
