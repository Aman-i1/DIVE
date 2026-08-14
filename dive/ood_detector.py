"""Out-of-Distribution (OOD) & Anomaly Detector - `dive/ood_detector.py`.

Monitors inference-time inputs against baseline training feature distributions using
multivariate distance (Mahalanobis / PCA reconstruction error) and Isolation Forest density.
Flags anomalous or out-of-distribution inputs with risk scores (SAFE, LOW_CONFIDENCE, OOD).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class OODResult:
    """OOD detection evaluation output."""

    ood_scores: np.ndarray  # 0.0 (in-distribution) to 1.0 (highly anomalous)
    status_labels: List[str]  # SAFE, LOW_CONFIDENCE, OOD
    mean_ood_score: float
    pct_ood: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_ood_score": round(self.mean_ood_score, 4),
            "pct_ood": round(self.pct_ood, 2),
            "ood_count": int(sum(1 for s in self.status_labels if s == "OOD")),
            "safe_count": int(sum(1 for s in self.status_labels if s == "SAFE")),
        }


class OODDetector:
    """Detects inference-time out-of-distribution inputs."""

    def __init__(self, contamination: float = 0.05, n_components: int = 5) -> None:
        self.contamination = contamination
        self.n_components = n_components
        self.scaler_ = StandardScaler()
        self.pca_: Optional[PCA] = None
        self.iso_forest_ = IsolationForest(contamination=contamination, random_state=42)
        self.mean_vector_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> "OODDetector":
        """Learn feature distribution on training dataset."""
        numeric_df = X.select_dtypes(include=[np.number]).fillna(0.0)
        self.feature_names_ = list(numeric_df.columns)

        if numeric_df.empty:
            self.is_fitted_ = True
            return self

        X_mat = numeric_df.to_numpy()
        X_scaled = self.scaler_.fit_transform(X_mat)

        n_comp = min(self.n_components, X_scaled.shape[1], max(1, X_scaled.shape[0] - 1))
        self.pca_ = PCA(n_components=n_comp)
        X_pca = self.pca_.fit_transform(X_scaled)

        # Fit Isolation Forest
        self.iso_forest_.fit(X_pca)

        # Compute empirical covariance and mean in PCA space for Mahalanobis metric
        self.mean_vector_ = np.mean(X_pca, axis=0)
        cov = np.cov(X_pca, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-4  # Regularization for numerical stability
        try:
            self.cov_inv_ = np.linalg.pinv(cov_reg)
        except Exception:
            self.cov_inv_ = np.eye(cov.shape[0])

        self.is_fitted_ = True
        return self

    def score(self, X: pd.DataFrame) -> OODResult:
        """Score rows in X and classify into SAFE, LOW_CONFIDENCE, or OOD."""
        if not self.is_fitted_:
            raise ValueError("OODDetector must be fitted with fit() before scoring.")

        if not self.feature_names_:
            n_rows = len(X)
            return OODResult(
                ood_scores=np.zeros(n_rows),
                status_labels=["SAFE"] * n_rows,
                mean_ood_score=0.0,
                pct_ood=0.0,
            )

        numeric_df = X.reindex(columns=self.feature_names_).select_dtypes(include=[np.number]).fillna(0.0)
        X_scaled = self.scaler_.transform(numeric_df.to_numpy())
        X_pca = self.pca_.transform(X_scaled) if self.pca_ is not None else X_scaled

        # 1. Isolation Forest anomaly scores (mapped 0.0 to 1.0)
        iso_raw = -self.iso_forest_.score_samples(X_pca)  # higher = more anomalous
        iso_scores = (iso_raw - iso_raw.min()) / max(iso_raw.max() - iso_raw.min(), 1e-6)

        # 2. Mahalanobis distance scores
        diff = X_pca - self.mean_vector_
        if self.cov_inv_ is not None:
            dist_sq = np.sum((diff @ self.cov_inv_) * diff, axis=1)
            mahal_dist = np.sqrt(np.maximum(dist_sq, 0.0))
            mahal_scores = mahal_dist / (np.median(mahal_dist) * 3.0 + 1e-6)
            mahal_scores = np.clip(mahal_scores, 0.0, 1.0)
        else:
            mahal_scores = iso_scores

        # Ensemble composite OOD score
        composite_scores = 0.5 * iso_scores + 0.5 * mahal_scores
        composite_scores = np.clip(composite_scores, 0.0, 1.0)

        status_labels: List[str] = []
        for s in composite_scores:
            if s > 0.75:
                status_labels.append("OOD")
            elif s > 0.50:
                status_labels.append("LOW_CONFIDENCE")
            else:
                status_labels.append("SAFE")

        mean_score = float(np.mean(composite_scores))
        pct_ood = float(np.mean([1 if s == "OOD" else 0 for s in status_labels]) * 100.0)

        return OODResult(
            ood_scores=composite_scores,
            status_labels=status_labels,
            mean_ood_score=mean_score,
            pct_ood=pct_ood,
        )
