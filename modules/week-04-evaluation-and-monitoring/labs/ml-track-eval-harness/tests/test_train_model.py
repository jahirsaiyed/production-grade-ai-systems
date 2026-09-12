import json

from train_model import main


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
