"""Transformer Configuration & Architecture Definitions - `dive/nlp/transformers/config.py`.

Provides configuration parameters and model registry mappings for Hugging Face Transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


TRANSFORMER_MODELS: Dict[str, str] = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-small",
}


@dataclass
class TransformerConfig:
    """Configuration for Hugging Face Transformer fine-tuning and inference."""

    model_name: str = "distilbert-base-uncased"
    max_seq_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    device: Optional[str] = None
    seed: int = 42

    def resolve_model_id(self) -> str:
        """Resolve short architecture alias (e.g. 'distilbert') to full Hugging Face model ID."""
        key = self.model_name.lower().strip()
        return TRANSFORMER_MODELS.get(key, self.model_name)
