import json

from train_model import HELD_OUT_SPLIT_RANDOM_STATE, main


def test_main_writes_model_manifest_and_held_out_split(tmp_path):
    main(artifact_dir=tmp_path)

    model_path = tmp_path / "model.joblib"
    manifest_path = tmp_path / "manifest.json"
    held_out_path = tmp_path / "held_out.json"

    assert model_path.exists()
    assert manifest_path.exists()
    assert held_out_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifact_version"] == "0.1.0"
    assert len(manifest["sha256"]) == 64
    assert "scikit-learn" in manifest["key_dependencies"]
    assert "created_at" in manifest
    assert "git_commit" in manifest

    held_out = json.loads(held_out_path.read_text())
    assert "X" in held_out
    assert "y" in held_out
    # 20% of the 2000-row synthetic dataset.
    assert len(held_out["y"]) == 400
    assert len(held_out["X"]) == len(held_out["y"])
    assert all(len(row) == 6 for row in held_out["X"])
    assert set(held_out["y"]) <= {0, 1}


def test_main_fits_only_on_the_train_split_not_the_full_dataset(tmp_path):
    from sklearn.linear_model import LogisticRegression
    import joblib
    import numpy as np

    main(artifact_dir=tmp_path)

    committed_model = joblib.load(tmp_path / "model.joblib")
    held_out = json.loads((tmp_path / "held_out.json").read_text())
    X_holdout = np.array(held_out["X"])

    # Regenerate the exact same full dataset + split train_model.py itself uses,
    # then refit on ONLY the train portion — this should match the committed
    # model's coefficients if (and only if) main() genuinely fit on the train
    # split rather than the full dataset.
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(
        n_samples=2000, n_features=6, n_informative=4,
        weights=[0.9, 0.1], random_state=42,
    )
    X_train, X_holdout_expected, y_train, y_holdout_expected = train_test_split(
        X, y, test_size=0.2, random_state=HELD_OUT_SPLIT_RANDOM_STATE, stratify=y
    )

    train_only_refit = LogisticRegression(max_iter=1000)
    train_only_refit.fit(X_train, y_train)

    assert np.allclose(committed_model.coef_, train_only_refit.coef_)

    # Zero row overlap between the held-out set and the training data — this
    # is what "held-out" actually means.
    train_rows = {tuple(row) for row in X_train}
    holdout_rows = {tuple(row) for row in X_holdout}
    assert train_rows.isdisjoint(holdout_rows)
