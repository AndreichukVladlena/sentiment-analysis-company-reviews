"""Добавить финальный локальный чекпойнт DeBERTa в существующий каталог API."""

import argparse
import json
import shutil
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory

REQUIRED_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "spm.model",
    "training_config.json",
}
OPTIONAL_FILES = {"special_tokens_map.json", "added_tokens.json"}


def export_deberta(checkpoint, output=Path("models")):
    checkpoint, output = Path(checkpoint), Path(output)
    if not all((checkpoint / name).is_file() for name in REQUIRED_FILES):
        raise ValueError("В чекпойнте отсутствуют обязательные файлы")
    config = json.loads(
        (checkpoint / "training_config.json").read_text(encoding="utf-8")
    )
    if (
        config.get("training_scope") != "full"
        or config.get("max_length") != 128
        or config.get("use_fast_tokenizer") is not False
    ):
        raise ValueError(
            "Нужен full-чекпойнт с max_length=128 и медленным токенизатором"
        )

    manifest_path = output / "manifest.json"
    catalog = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = output / "deberta"
    allowed = REQUIRED_FILES | OPTIONAL_FILES
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("Каталог назначения должен быть обычной директорией")
    existing = list(destination.iterdir()) if destination.exists() else []
    if any(
        path.name not in allowed or not path.is_file() or path.is_symlink()
        for path in existing
    ):
        raise ValueError("В каталоге назначения есть посторонние файлы")
    names = sorted(
        REQUIRED_FILES
        | {name for name in OPTIONAL_FILES if (checkpoint / name).is_file()}
    )

    # Copy and verify all files before replacing the existing weights.
    with TemporaryDirectory(prefix=".deberta-export-", dir=output) as temporary:
        staging = Path(temporary)
        hashes = {}
        for name in names:
            shutil.copyfile(checkpoint / name, staging / name)
            with (staging / name).open("rb") as stream:
                hashes[name] = file_digest(stream, "sha256").hexdigest()
        spec = {
            "id": "deberta",
            "format": "deberta",
            "filename": "deberta",
            "version": f"full-{hashes['model.safetensors'][:12]}",
            "description": "DeBERTa-v3-base: финальное обучение на всех данных, 128 токенов",
            "file_sha256": hashes,
        }
        catalog["models"] = [
            entry for entry in catalog["models"] if entry["id"] != "deberta"
        ] + [spec]
        (staging / "manifest.json").write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        destination.mkdir(exist_ok=True)
        for name in names:
            (staging / name).replace(destination / name)
        for path in existing:
            if path.name not in hashes:
                path.unlink()
        (staging / "manifest.json").replace(manifest_path)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="Каталог финального save_pretrained"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Каталог с существующим manifest.json (по умолчанию models)",
    )
    args = parser.parse_args()
    try:
        destination = export_deberta(args.checkpoint, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Финальная DeBERTa добавлена в каталог API: {destination}")


if __name__ == "__main__":
    main()
