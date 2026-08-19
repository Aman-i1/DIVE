"""Deterministic NLP Dataset Splitting - `dive/nlp/data/splitter.py`.

Provides reproducible splitting utilities for cross-validation and holdout validation.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from dive.nlp.data.dataset import NLPDataset
from dive.nlp.exceptions import TextDataError


class DatasetSplitter:
    """Deterministic splitter for NLP datasets."""

    def __init__(
        self,
        strategy: str = "stratified_kfold",
        cv_splits: int = 5,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42,
    ) -> None:
        self.strategy = strategy
        self.cv_splits = cv_splits
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state

    def train_test_split(
        self, dataset: NLPDataset, val_size: Optional[float] = None
    ) -> Tuple[NLPDataset, ...]:
        """Perform deterministic train/test or train/val/test split."""
        return dataset.split(
            test_size=self.test_size,
            val_size=val_size,
            stratify=self.stratify,
            random_state=self.random_state,
        )

    def cv_folds(self, dataset: NLPDataset) -> Iterator[Tuple[NLPDataset, NLPDataset]]:
        """Yield (train_dataset, val_dataset) pairs for cross-validation."""
        n_samples = len(dataset)
        if n_samples < self.cv_splits:
            raise TextDataError(
                f"Cannot perform {self.cv_splits}-fold CV on a dataset with only {n_samples} sample(s)."
            )

        labels = dataset.labels
        use_stratify = self.stratify and labels is not None and len(set(labels)) > 1

        if use_stratify:
            skf = StratifiedKFold(
                n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
            )
            split_iter = skf.split(dataset.texts, labels)
        else:
            kf = KFold(
                n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
            )
            split_iter = kf.split(dataset.texts)

        for fold_idx, (train_indices, val_indices) in enumerate(split_iter):
            train_ds = dataset.subset(train_indices, name=f"{dataset.name}_fold{fold_idx}_train")
            val_ds = dataset.subset(val_indices, name=f"{dataset.name}_fold{fold_idx}_val")
            yield train_ds, val_ds
