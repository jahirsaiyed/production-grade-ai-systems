import json

import numpy as np
import pytest

from app.adapters.index_store import ArtifactIntegrityError, _sha256_of, load_index


def _write_artifact(tmp_path):
    payload = {
        "chunks": [{"text": "hello world", "source": "doc.md", "chunk_id": 0}],
        "dense_vectors": [[0.1, 0.2, 0.3]],
        "bm25_tokenized_corpus": [["hello", "world"]],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(payload))

    manifest = {
        "artifact_version": "0.1.0",
        "sha256": _sha256_of(index_path),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_load_index_succeeds_when_hash_matches(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    loaded = load_index(artifact_dir)
    assert loaded.chunks[0].source == "doc.md"
    assert np.array_equal(loaded.dense_vectors[0], np.array([0.1, 0.2, 0.3]))
    assert loaded.bm25_tokenized_corpus == [["hello", "world"]]


def test_load_index_raises_when_hash_mismatches(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactIntegrityError):
        load_index(artifact_dir)
