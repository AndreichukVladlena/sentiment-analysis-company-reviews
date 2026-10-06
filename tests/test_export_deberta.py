"""Экспорт принимает только полный совместимый чекпойнт и сохраняет каталог."""

import json
import subprocess
import sys

import pytest


@pytest.fixture
def export_paths(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "spm.model",
    ]:
        (checkpoint / name).write_bytes(b"test artifact")
    (checkpoint / "training_config.json").write_text(
        json.dumps(
            {
                "training_scope": "full",
                "max_length": 128,
                "use_fast_tokenizer": False,
            }
        )
    )
    output = tmp_path / "models"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "default_model": "tfidf",
                "models": [{"id": "tfidf", "version": "preserved"}],
            }
        )
    )
    return checkpoint, output


@pytest.mark.parametrize("problem", ["missing_file", "fold", "length", "tokenizer"])
def test_invalid_checkpoint_preserves_manifest_and_existing_files(
    export_paths, problem
):
    from company_reviews.export_deberta import export_deberta

    checkpoint, output = export_paths
    old = output / "deberta"
    old.mkdir()
    (old / "model.safetensors").write_bytes(b"previous weights")
    manifest = (output / "manifest.json").read_bytes()
    metadata = checkpoint / "training_config.json"
    config = json.loads(metadata.read_text())
    if problem == "missing_file":
        (checkpoint / "spm.model").unlink()
    elif problem == "fold":
        config.pop("training_scope")
    elif problem == "length":
        config["max_length"] = 256
    else:
        config["use_fast_tokenizer"] = True
    metadata.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        export_deberta(checkpoint, output)
    assert (output / "manifest.json").read_bytes() == manifest
    assert (old / "model.safetensors").read_bytes() == b"previous weights"


def test_export_rejects_unrelated_destination_files(export_paths):
    from company_reviews.export_deberta import export_deberta

    checkpoint, output = export_paths
    destination = output / "deberta"
    destination.mkdir()
    (destination / "notes.txt").write_text("keep me")
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="посторонние"):
        export_deberta(checkpoint, output)
    assert (destination / "notes.txt").read_text() == "keep me"
    assert (output / "manifest.json").read_bytes() == before


def test_reexport_removes_obsolete_optional_files_and_keeps_other_models(export_paths):
    from company_reviews.export_deberta import export_deberta

    checkpoint, output = export_paths
    (checkpoint / "added_tokens.json").write_text("{}")
    (checkpoint / "valid_probabilities_test.npy").write_bytes(b"do not export")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "company_reviews.export_deberta",
            str(checkpoint),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    first = json.loads((output / "manifest.json").read_text())
    assert first["default_model"] == "tfidf"
    assert first["models"][0] == {"id": "tfidf", "version": "preserved"}
    assert first["models"][1]["version"].startswith("full-")
    assert not (output / "deberta/valid_probabilities_test.npy").exists()
    (checkpoint / "added_tokens.json").unlink()
    export_deberta(checkpoint, output)
    updated = json.loads((output / "manifest.json").read_text())
    assert [entry["id"] for entry in updated["models"]] == ["tfidf", "deberta"]
    assert not (output / "deberta/added_tokens.json").exists()
    assert "added_tokens.json" not in updated["models"][1]["file_sha256"]
