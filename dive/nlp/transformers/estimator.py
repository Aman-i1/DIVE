"""Deep Pretrained Transformer Estimators - `dive/nlp/transformers/estimator.py`.

Implements fine-tuning and inference for sequence classification and regression
using Hugging Face Transformers and PyTorch, with optional dependency safety.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder

from dive.nlp.features.tfidf import TFIDFRepresentation
from dive.nlp.interfaces import NLPEstimatorProtocol
from dive.nlp.transformers.config import TransformerConfig
from dive.utils.optional import is_available, load_optional

logger = logging.getLogger(__name__)


def _detect_torch_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class TransformerClassifier:
    """Sequence classification estimator conforming to NLPEstimatorProtocol."""

    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        model_name: str = "distilbert",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_seq_length: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or TransformerConfig(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            device=device,
        )
        self.model_id = self.config.resolve_model_id()
        self.device = self.config.device or _detect_torch_device()

        self.classes_: Optional[np.ndarray] = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_fallback: bool = False
        self._fallback_pipeline: Any = None
        self.fitted_ = False

    def _init_backend(self, num_labels: int) -> bool:
        """Attempt to load Hugging Face tokenizer and model."""
        if is_available("transformers") and is_available("torch"):
            try:
                transformers = load_optional("transformers", purpose="Transformer modeling")
                AutoTokenizer = getattr(transformers, "AutoTokenizer")
                AutoModelForSequenceClassification = getattr(
                    transformers, "AutoModelForSequenceClassification"
                )

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id, num_labels=num_labels
                )
                self._model.to(self.device)
                self._is_fallback = False
                return True
            except Exception as e:
                logger.warning(
                    f"Could not load Hugging Face model '{self.model_id}': {e}. "
                    "Using fast CPU fallback classifier."
                )

        self._is_fallback = True
        return False

    def fit(
        self, texts: Sequence[str], y: Sequence[Any]
    ) -> "TransformerClassifier":
        """Fine-tune the Transformer sequence classifier."""
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        num_labels = len(self.classes_)

        has_hf = self._init_backend(num_labels=num_labels)

        if has_hf:
            try:
                import torch
                from torch.utils.data import DataLoader, TensorDataset

                # Map labels to 0..num_labels-1
                label_map = {cls_name: i for i, cls_name in enumerate(self.classes_)}
                y_indices = torch.tensor([label_map[val] for val in y_arr], dtype=torch.long)

                # Tokenize batch
                encoded = self._tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                dataset = TensorDataset(input_ids, attention_mask, y_indices)
                loader = DataLoader(
                    dataset,
                    batch_size=min(self.config.batch_size, len(texts)),
                    shuffle=True,
                )

                optimizer = torch.optim.AdamW(
                    self._model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )

                self._model.train()
                for _ in range(self.config.epochs):
                    for batch_ids, batch_mask, batch_labels in loader:
                        batch_ids = batch_ids.to(self.device)
                        batch_mask = batch_mask.to(self.device)
                        batch_labels = batch_labels.to(self.device)

                        optimizer.zero_grad()
                        outputs = self._model(
                            input_ids=batch_ids,
                            attention_mask=batch_mask,
                            labels=batch_labels,
                        )
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()

                self.fitted_ = True
                return self
            except Exception as e:
                logger.warning(f"Transformer fine-tuning failed: {e}. Switching to CPU fallback.")
                self._is_fallback = True

        # Fallback path for environments without PyTorch/Transformers
        self._fallback_rep = TFIDFRepresentation(min_df=1)
        X_feats = self._fallback_rep.fit_transform(texts)
        self._fallback_model = LogisticRegression(class_weight="balanced", max_iter=1000)
        self._fallback_model.fit(X_feats, y_arr)
        self.fitted_ = True
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Generate class probability distributions."""
        if not self.fitted_:
            raise RuntimeError("TransformerClassifier is not fitted. Call .fit() first.")

        if self._is_fallback:
            X_feats = self._fallback_rep.transform(texts)
            return self._fallback_model.predict_proba(X_feats)

        import torch
        self._model.eval()
        with torch.no_grad():
            encoded = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probas = torch.softmax(logits, dim=-1).cpu().numpy()
            return probas

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Generate class predictions."""
        probas = self.predict_proba(texts)
        pred_indices = np.argmax(probas, axis=1)
        return self.classes_[pred_indices]


class TransformerRegressor:
    """Continuous sequence regression estimator conforming to NLPEstimatorProtocol."""

    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        model_name: str = "distilbert",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_seq_length: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or TransformerConfig(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            device=device,
        )
        self.model_id = self.config.resolve_model_id()
        self.device = self.config.device or _detect_torch_device()

        self._model: Any = None
        self._tokenizer: Any = None
        self._is_fallback: bool = False
        self.fitted_ = False

    def _init_backend(self) -> bool:
        if is_available("transformers") and is_available("torch"):
            try:
                transformers = load_optional("transformers", purpose="Transformer regression")
                AutoTokenizer = getattr(transformers, "AutoTokenizer")
                AutoModelForSequenceClassification = getattr(
                    transformers, "AutoModelForSequenceClassification"
                )

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id, num_labels=1
                )
                self._model.to(self.device)
                self._is_fallback = False
                return True
            except Exception as e:
                logger.warning(
                    f"Could not load Hugging Face regression model '{self.model_id}': {e}. "
                    "Using fast CPU fallback regressor."
                )

        self._is_fallback = True
        return False

    def fit(
        self, texts: Sequence[str], y: Sequence[Any]
    ) -> "TransformerRegressor":
        """Fine-tune Transformer regressor."""
        y_arr = np.asarray(y, dtype=np.float32)
        has_hf = self._init_backend()

        if has_hf:
            try:
                import torch
                from torch.utils.data import DataLoader, TensorDataset

                y_targets = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
                encoded = self._tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                dataset = TensorDataset(input_ids, attention_mask, y_targets)
                loader = DataLoader(
                    dataset,
                    batch_size=min(self.config.batch_size, len(texts)),
                    shuffle=True,
                )

                optimizer = torch.optim.AdamW(
                    self._model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )

                self._model.train()
                for _ in range(self.config.epochs):
                    for batch_ids, batch_mask, batch_targets in loader:
                        batch_ids = batch_ids.to(self.device)
                        batch_mask = batch_mask.to(self.device)
                        batch_targets = batch_targets.to(self.device)

                        optimizer.zero_grad()
                        outputs = self._model(
                            input_ids=batch_ids,
                            attention_mask=batch_mask,
                            labels=batch_targets,
                        )
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()

                self.fitted_ = True
                return self
            except Exception as e:
                logger.warning(f"Transformer fine-tuning failed: {e}. Switching to CPU fallback.")
                self._is_fallback = True

        self._fallback_rep = TFIDFRepresentation(min_df=1)
        X_feats = self._fallback_rep.fit_transform(texts)
        self._fallback_model = Ridge(alpha=1.0)
        self._fallback_model.fit(X_feats, y_arr)
        self.fitted_ = True
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Generate continuous regression predictions."""
        if not self.fitted_:
            raise RuntimeError("TransformerRegressor is not fitted. Call .fit() first.")

        if self._is_fallback:
            X_feats = self._fallback_rep.transform(texts)
            return self._fallback_model.predict(X_feats)

        import torch
        self._model.eval()
        with torch.no_grad():
            encoded = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(1).cpu().numpy()
            return logits
