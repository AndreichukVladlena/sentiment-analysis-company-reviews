"""CPU-инференс DeBERTa из локального каталога save_pretrained."""

from collections.abc import Sequence
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer


class DebertaClassifier:
    def __init__(self, directory: Path):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(
            directory, local_files_only=True
        )
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            directory, local_files_only=True, use_safetensors=True
        ).to("cpu")
        if self.model.config.num_labels != 5:
            raise ValueError("DeBERTa должна возвращать пять оценок")
        self.model.eval()
        self.classes_ = np.arange(1, 6)
        # Concurrent HTTP requests must not multiply memory usage for inference batches.
        self._inference_lock = Lock()

    def predict_proba(self, reviews: Sequence[str]) -> np.ndarray:
        probabilities = []
        with self._inference_lock, torch.inference_mode():
            for start in range(0, len(reviews), 8):
                inputs = self.tokenizer(
                    list(reviews[start : start + 8]),
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                logits = self.model(**inputs).logits
                probabilities.append(logits.softmax(dim=-1).numpy())
        return np.concatenate(probabilities)
