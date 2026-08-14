"""Diversity-Aware Ensembling & Error Correlation Selector - `dive/ensemble_diversity.py`.

Evaluates pairwise model prediction correlation, Yule's Q-statistic, and disagreement
metrics to construct ensembles of complementary models with uncorrelated failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from dive.decisions import DecisionLogger


@dataclass
class DiversityMatrix:
    """Pairwise diversity and error correlation metrics across candidate models."""

    model_names: List[str]
    correlation_matrix: np.ndarray  # Pearson correlation of predictions
    q_statistics: np.ndarray  # Yule's Q statistic (-1 = high diversity, +1 = redundant)
    disagreement_rates: np.ndarray  # Fraction of samples where models disagree

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_names": self.model_names,
            "correlation_matrix": self.correlation_matrix.round(4).tolist(),
            "q_statistics": self.q_statistics.round(4).tolist(),
            "disagreement_rates": self.disagreement_rates.round(4).tolist(),
        }


class ModelDiversityEvaluator:
    """Selects complementary model subsets maximizing ensemble performance while penalizing redundancy."""

    def __init__(self, problem_type: str = "classification", logger: Optional[DecisionLogger] = None) -> None:
        self.problem_type = problem_type
        self.logger = logger or DecisionLogger()

    def compute_diversity_matrix(
        self,
        predictions_dict: Dict[str, np.ndarray],  # model_name -> 1D prediction array
        y_true: np.ndarray,
    ) -> DiversityMatrix:
        """Compute pairwise prediction correlation, Q-statistic, and disagreement matrix."""
        model_names = list(predictions_dict.keys())
        k = len(model_names)
        y_true = np.asarray(y_true)

        corr_mat = np.eye(k)
        q_mat = np.eye(k)
        disagree_mat = np.zeros((k, k))

        preds_matrix = np.column_stack([predictions_dict[name] for name in model_names])

        # 1. Pearson correlation matrix of predictions
        if preds_matrix.shape[0] > 1:
            corr_mat = np.corrcoef(preds_matrix, rowvar=False)
            corr_mat = np.nan_to_num(corr_mat, nan=1.0)

        # 2. Pairwise Q-statistic and Disagreement rate
        for i in range(k):
            for j in range(i + 1, k):
                p_i = predictions_dict[model_names[i]]
                p_j = predictions_dict[model_names[j]]

                if self.problem_type == "classification":
                    correct_i = (p_i == y_true)
                    correct_j = (p_j == y_true)

                    n11 = np.sum(correct_i & correct_j)  # both correct
                    n00 = np.sum((~correct_i) & (~correct_j))  # both wrong
                    n10 = np.sum(correct_i & (~correct_j))  # i correct, j wrong
                    n01 = np.sum((~correct_i) & correct_j)  # i wrong, j correct

                    denom = (n11 * n00 + n10 * n01)
                    q_val = (n11 * n00 - n10 * n01) / max(denom, 1e-6)
                    disagree_val = float(np.mean(p_i != p_j))
                else:
                    # For regression, residuals
                    res_i = np.abs(p_i - y_true)
                    res_j = np.abs(p_j - y_true)
                    q_val = float(np.corrcoef(res_i, res_j)[0, 1]) if len(res_i) > 1 else 1.0
                    q_val = np.nan_to_num(q_val, nan=1.0)
                    disagree_val = float(np.mean(np.abs(p_i - p_j) / (np.std(y_true) + 1e-6)))

                q_mat[i, j] = q_mat[j, i] = float(np.clip(q_val, -1.0, 1.0))
                disagree_mat[i, j] = disagree_mat[j, i] = float(disagree_val)

        return DiversityMatrix(
            model_names=model_names,
            correlation_matrix=corr_mat,
            q_statistics=q_mat,
            disagreement_rates=disagree_mat,
        )

    def select_diverse_subset(
        self,
        predictions_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
        max_models: int = 4,
        diversity_weight: float = 0.25,
    ) -> List[str]:
        """Greedy forward selection with diversity penalty to select optimal ensemble members."""
        model_names = list(predictions_dict.keys())
        if len(model_names) <= max_models:
            return model_names

        div_matrix = self.compute_diversity_matrix(predictions_dict, y_true)
        y_true = np.asarray(y_true)

        # Evaluate individual model scores
        scores: Dict[str, float] = {}
        for name in model_names:
            preds = predictions_dict[name]
            if self.problem_type == "classification":
                scores[name] = float(accuracy_score(y_true, preds))
            else:
                scores[name] = float(max(0.0, r2_score(y_true, preds)))

        # 1. Start with the single best model
        best_first = max(scores.keys(), key=lambda k: scores[k])
        selected = [best_first]
        remaining = [m for m in model_names if m != best_first]

        # 2. Greedily add models maximizing combined score - diversity penalty
        while len(selected) < max_models and remaining:
            best_candidate = None
            best_composite = -float("inf")

            for cand in remaining:
                cand_score = scores[cand]
                cand_idx = model_names.index(cand)

                # Measure average correlation with already selected models
                corrs = [div_matrix.correlation_matrix[cand_idx, model_names.index(s)] for s in selected]
                avg_corr = float(np.mean(corrs))

                # Composite metric: high predictive score with low correlation penalty
                composite = cand_score - (diversity_weight * avg_corr)

                if composite > best_composite:
                    best_composite = composite
                    best_candidate = cand

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        self.logger.log(
            component="EnsembleDiversityEngine",
            decision=f"Selected {len(selected)} diverse models for ensemble: {', '.join(selected)}",
            reason="Greedy forward selection with correlation penalty",
            confidence=0.95,
            evidence={"selected_models": selected, "total_candidates": len(model_names)},
        )

        return selected
