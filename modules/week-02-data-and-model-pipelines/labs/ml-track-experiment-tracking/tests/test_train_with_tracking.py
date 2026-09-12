import json

from train_with_tracking import main


def test_main_logs_expected_metrics_and_writes_manifest(tmp_path):
    manifest_path = tmp_path / "reproducibility_manifest.json"
    tracking_uri = f"file:{tmp_path / 'mlruns'}"

    manifest = main(tracking_uri=tracking_uri, manifest_path=manifest_path)

    assert manifest_path.exists()
    written = json.loads(manifest_path.read_text())
    assert written["mlflow_run_id"] == manifest["mlflow_run_id"]
    assert "accuracy" in manifest["metrics"]
    assert "f1" in manifest["metrics"]
    assert 0.0 <= manifest["metrics"]["accuracy"] <= 1.0
    assert len(manifest["data_digest"]) == 64
    assert len(manifest["env_lock"]) == 64
    assert manifest["artifact_uri"].startswith(tracking_uri)
