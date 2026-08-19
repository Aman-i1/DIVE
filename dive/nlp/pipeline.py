"""End-to-End NLP Pipeline - `dive/nlp/pipeline.py`.

Orchestrates preprocessing, feature extraction, and estimator inference
in a unified, self-contained pipeline object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dive.nlp.features.tfidf import TFIDFRepresentation
from dive.nlp.interfaces import NLPPipelineProtocol
from dive.nlp.preprocessing.preprocessor import NLPPreprocessor


class NLPPipeline:
    """Complete NLP Pipeline implementing NLPPipelineProtocol."""

    def __init__(
        self,
        estimator: Any,
        preprocessor: Optional[NLPPreprocessor] = None,
        representation: Optional[Any] = None,
        task_type: str = "text_classification",
        model_name: str = "NLPPipeline",
    ) -> None:
        self.estimator = estimator
        self.preprocessor = preprocessor or NLPPreprocessor()
        self.representation = representation or TFIDFRepresentation()
        self.task_type = task_type
        self.model_name = model_name

        self.label_encoder_: Optional[LabelEncoder] = None
        self.label_lookup_: Dict[str, Any] = {}
        self.fitted_ = False

    def fit(self, texts: Sequence[str], y: Sequence[Any]) -> "NLPPipeline":
        """Fit preprocessing, representation, and estimator on training text and targets."""
        # 1. Target encoding if classification
        if self.task_type in ("text_classification", "classification"):
            self.label_encoder_ = LabelEncoder()
            y_raw = list(y)
            y_encoded = self.label_encoder_.fit_transform([str(val) for val in y_raw])
            self.label_lookup_ = {str(val): val for val in y_raw}
            y_target = y_encoded
        else:
            y_target = np.asarray(y, dtype=np.float32)

        # 2. Preprocess text
        cleaned_texts = self.preprocessor.fit_transform(texts)

        # 3. Feature representation
        X_feats = self.representation.fit_transform(cleaned_texts)

        # 4. Train estimator
        self.estimator.fit(X_feats, y_target)
        self.fitted_ = True
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Generate predictions for a sequence of texts."""
        if not self.fitted_:
            raise RuntimeError("NLPPipeline is not fitted. Call .fit() first.")

        cleaned = self.preprocessor.transform(texts)
        X_feats = self.representation.transform(cleaned)
        preds = self.estimator.predict(X_feats)

        # Decode labels back to original format
        if self.label_encoder_ is not None and self.label_lookup_:
            decoded = [self.label_lookup_.get(str(cls_name), cls_name) for cls_name in self.label_encoder_.inverse_transform(preds)]
            return np.array(decoded, dtype=object)

        return preds

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Generate class probability distributions."""
        if not self.fitted_:
            raise RuntimeError("NLPPipeline is not fitted. Call .fit() first.")

        if self.task_type not in ("text_classification", "classification"):
            raise ValueError("predict_proba is only supported for classification tasks.")

        cleaned = self.preprocessor.transform(texts)
        X_feats = self.representation.transform(cleaned)

        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X_feats)

        raise AttributeError(f"Underlying estimator '{type(self.estimator).__name__}' does not support predict_proba.")

    @property
    def class_names(self) -> Optional[List[str]]:
        """Return list of human-readable class names."""
        if self.label_encoder_ is not None:
            return [str(self.label_lookup_.get(str(c), c)) for c in self.label_encoder_.classes_]
        return None
