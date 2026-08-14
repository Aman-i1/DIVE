"""Dataset Fingerprinting & Meta-Learning Engine - `dive/meta_learning.py`.

Extracts statistical, structural, information-theoretic, and landmarking meta-features
from datasets to match against meta-learning knowledge bases, recommending model family
priors, initial hyperparameter spaces, and expected search budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dive.decisions import DecisionLogger


@dataclass
class DatasetFingerprint:
    """Statistical, structural, and landmarking signature of a dataset."""

    dataset_hash: str
    n_samples: int
    n_features: int
    sample_to_feature_ratio: float
    n_numeric: int
    n_categorical: int
    sparsity: float
    mean_skewness: float
    mean_kurtosis: float
    target_entropy: float
    landmark_linear_score: float
    landmark_tree_stump_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_hash": self.dataset_hash,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "sample_to_feature_ratio": round(self.sample_to_feature_ratio, 2),
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "sparsity": round(self.sparsity, 4),
            "mean_skewness": round(self.mean_skewness, 4),
            "mean_kurtosis": round(self.mean_kurtosis, 4),
            "target_entropy": round(self.target_entropy, 4),
            "landmark_linear_score": round(self.landmark_linear_score, 4),
            "landmark_tree_stump_score": round(self.landmark_tree_stump_score, 4),
        }


@dataclass
class MetaWarmStartPriors:
    """Warm-start priors recommended by the meta-learning engine."""

    recommended_model_families: List[str]
    suggested_initial_learning_rate: float
    suggested_max_depth: int
    suggested_search_budget_scale: float  # Multiplier on default search budget
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_model_families": self.recommended_model_families,
            "suggested_initial_learning_rate": self.suggested_initial_learning_rate,
            "suggested_max_depth": self.suggested_max_depth,
            "suggested_search_budget_scale": self.suggested_search_budget_scale,
            "rationale": self.rationale,
        }


class MetaLearningEngine:
    """Computes dataset fingerprints and provides warm-start optimization priors."""

    def __init__(self, logger: Optional[DecisionLogger] = None) -> None:
        self.logger = logger or DecisionLogger()

    def compute_fingerprint(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str = "classification",
    ) -> DatasetFingerprint:
        """Extract statistical and landmarking meta-features from dataset."""
        n, p = X.shape
        numeric_df = X.select_dtypes(include=[np.number])
        n_num = numeric_df.shape[1]
        n_cat = p - n_num

        # Hash dataset structure & sample values
        hash_input = f"{n}_{p}_{list(X.columns)}_{X.head(5).to_dict()}".encode("utf-8")
        dataset_hash = hashlib.sha256(hash_input).hexdigest()[:12]

        sparsity = float(X.isnull().to_numpy().mean()) if p > 0 else 0.0

        # Statistical moments
        if n_num > 0 and n > 2:
            skew_vals = numeric_df.skew().dropna()
            mean_skew = float(skew_vals.mean()) if not skew_vals.empty else 0.0

            kurt_vals = numeric_df.kurt().dropna()
            mean_kurt = float(kurt_vals.mean()) if not kurt_vals.empty else 0.0
        else:
            mean_skew = 0.0
            mean_kurt = 0.0

        # Target entropy
        if problem_type == "classification":
            probs = y.value_counts(normalize=True).to_numpy()
            probs = probs[probs > 0]
            target_entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            std_y = float(y.std()) if len(y) > 1 else 1.0
            target_entropy = float(0.5 * np.log(2 * np.pi * np.e * (std_y**2 + 1e-8)))

        # Landmarkers: fast baseline models on filled numeric subset
        if n_num > 0 and n >= 6:
            X_filled = numeric_df.fillna(numeric_df.median())
            # 1. Linear landmark
            try:
                if problem_type == "classification":
                    lin_model = RidgeClassifier(alpha=1.0)
                    lin_model.fit(X_filled, y)
                    landmark_linear = float(lin_model.score(X_filled, y))
                else:
                    lin_reg = Ridge(alpha=1.0)
                    lin_reg.fit(X_filled, y)
                    landmark_linear = float(max(0.0, lin_reg.score(X_filled, y)))
            except Exception:
                landmark_linear = 0.5

            # 2. Decision Stump landmark (depth=1 tree)
            try:
                if problem_type == "classification":
                    tree_model = DecisionTreeClassifier(max_depth=1, random_state=42)
                    tree_model.fit(X_filled, y)
                    landmark_tree = float(tree_model.score(X_filled, y))
                else:
                    tree_reg = DecisionTreeRegressor(max_depth=1, random_state=42)
                    tree_reg.fit(X_filled, y)
                    landmark_tree = float(max(0.0, tree_reg.score(X_filled, y)))
            except Exception:
                landmark_tree = 0.5
        else:
            landmark_linear = 0.5
            landmark_tree = 0.5

        return DatasetFingerprint(
            dataset_hash=dataset_hash,
            n_samples=n,
            n_features=p,
            sample_to_feature_ratio=n / max(p, 1),
            n_numeric=n_num,
            n_categorical=n_cat,
            sparsity=sparsity,
            mean_skewness=mean_skew,
            mean_kurtosis=mean_kurt,
            target_entropy=target_entropy,
            landmark_linear_score=landmark_linear,
            landmark_tree_stump_score=landmark_tree,
        )

    def warm_start_recommendations(
        self,
        fingerprint: DatasetFingerprint,
        problem_type: str = "classification",
    ) -> MetaWarmStartPriors:
        """Infer model priors, learning rates, and tree depths from fingerprint."""
        # High sample to feature ratio + high tree landmark -> GBDTs & Random Forests
        if fingerprint.sample_to_feature_ratio > 50 and fingerprint.landmark_tree_stump_score >= fingerprint.landmark_linear_score:
            models = ["LightGBM", "XGBoost", "HistGradientBoosting", "RandomForest"]
            lr = 0.05
            depth = 6
            budget_scale = 1.0
            rationale = "High sample-to-feature ratio with strong tree-stump landmark signal indicates non-linear tree ensembles will dominate."
        elif fingerprint.sample_to_feature_ratio < 10 or fingerprint.landmark_linear_score > fingerprint.landmark_tree_stump_score:
            # Low sample count or strong linear signal -> Regularized Linear Models + shallow trees
            models = ["Ridge", "LogisticRegression" if problem_type == "classification" else "LinearRegression", "HistGradientBoosting"]
            lr = 0.02
            depth = 3
            budget_scale = 0.75
            rationale = "Low sample ratio or strong linear landmark signal suggests high risk of tree overfitting; prioritizing regularized linear and shallow models."
        else:
            models = ["HistGradientBoosting", "RandomForest", "ExtraTrees", "Ridge"]
            lr = 0.03
            depth = 5
            budget_scale = 1.0
            rationale = "Balanced dataset profile; deploying balanced multi-family search space."

        self.logger.log(
            component="MetaLearningEngine",
            decision=f"Warm-start model families: {', '.join(models)}",
            reason=rationale,
            confidence=0.92,
            evidence={
                "dataset_hash": fingerprint.dataset_hash,
                "n_samples": fingerprint.n_samples,
                "sample_to_feature_ratio": fingerprint.sample_to_feature_ratio,
                "linear_landmark": fingerprint.landmark_linear_score,
                "tree_landmark": fingerprint.landmark_tree_stump_score,
            },
        )

        return MetaWarmStartPriors(
            recommended_model_families=models,
            suggested_initial_learning_rate=lr,
            suggested_max_depth=depth,
            suggested_search_budget_scale=budget_scale,
            rationale=rationale,
        )
