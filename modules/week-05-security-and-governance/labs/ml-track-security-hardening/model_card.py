"""
Generates a Markdown model card for the fraud-detection model, documenting
intended use, training data, and known limitations — following the "model
cards for model reporting" convention (Mitchell et al., 2019).

Run with `make model-card` or `python model_card.py`.
"""
from pathlib import Path

MODEL_NAME = "Fraud Detection Model"
MODEL_VERSION = "0.1.0"

INTENDED_USE = (
    "Flags transactions with a high probability of fraud for manual review. "
    "Not intended for fully automated blocking without human oversight."
)

TRAINING_DATA_DESCRIPTION = (
    "Synthetic data generated via scikit-learn's make_classification "
    "(2000 samples, 6 features, ~10% positive class, random_state=42). "
    "Does not reflect any real transaction data or real customer population."
)

LIMITATIONS = [
    "Trained on synthetic data — performance on real transactions is unverified.",
    "Precision/recall at the fixed 0.5 decision threshold are modest on held-out "
    "synthetic data; see this course's Week 4 evaluation-harness lab for measured numbers.",
    "No monitoring for data or concept drift is implemented in this lab.",
]

OUT_OF_SCOPE_USES = [
    "Fully automated transaction blocking without human review.",
    "Any use on real financial data without re-training and re-validation.",
]


def generate_model_card() -> str:
    limitations_md = "\n".join(f"- {item}" for item in LIMITATIONS)
    out_of_scope_md = "\n".join(f"- {item}" for item in OUT_OF_SCOPE_USES)
    return (
        f"# Model Card: {MODEL_NAME}\n\n"
        f"**Version:** {MODEL_VERSION}\n\n"
        f"## Intended Use\n{INTENDED_USE}\n\n"
        f"## Training Data\n{TRAINING_DATA_DESCRIPTION}\n\n"
        f"## Limitations\n{limitations_md}\n\n"
        f"## Out-of-Scope Uses\n{out_of_scope_md}\n"
    )


def main() -> None:
    card = generate_model_card()
    output_path = Path(__file__).parent / "MODEL_CARD.md"
    output_path.write_text(card, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
