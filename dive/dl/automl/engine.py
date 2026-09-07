"""AutoDL Autonomous Exploration and Model Selection Engine - ``dive/dl/automl/engine.py``."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from dive.dl.automl.leaderboard import DLLeaderboard
from dive.dl.automl.trial import DLTrial
from dive.dl.capability import assess_dl_workload
from dive.dl.config import DLConfig, DLDataConfig, DLTask, DLTrainingConfig, Modality
from dive.dl.core.trainer import DLTrainer
from dive.dl.exceptions import DLDataError, DLTrainingError
from dive.dl.inference.predictor import DLPredictor
from dive.dl.modalities.base import get_adapter
from dive.resources import REFUSE


class AutoDL:
    """Autonomous Deep Learning architecture exploration and model selection orchestrator."""

    def __init__(
        self,
        max_trials: int = 5,
        mode: str = "balanced",
        optimize_for: Optional[str] = None,
        time_budget_secs: float = 600.0,
        force_fallback: bool = False,
    ) -> None:
        self.max_trials = max(1, int(max_trials))
        self.mode = mode if mode in ("fast", "balanced", "quality") else "balanced"
        self.time_budget_secs = float(time_budget_secs)
        self.force_fallback = bool(force_fallback)
        self.optimize_for = optimize_for

    def _generate_candidate_configs(self, base_config: DLConfig) -> List[Dict[str, Any]]:
        """Generate architectural search candidates based on search budget."""
        architectures = [
            {"hidden_sizes": [256, 128], "lr": 1e-3, "dropout": 0.1},
            {"hidden_sizes": [128, 64], "lr": 2e-3, "dropout": 0.05},
            {"hidden_sizes": [512, 256], "lr": 5e-4, "dropout": 0.15},
            {"hidden_sizes": [256], "lr": 1e-3, "dropout": 0.1},
            {"hidden_sizes": [512, 256, 128], "lr": 3e-4, "dropout": 0.2},
            {"hidden_sizes": [64, 32], "lr": 3e-3, "dropout": 0.0},
        ]
        return architectures[: self.max_trials]

    def fit(
        self,
        data: Any,
        modality: Union[str, Modality] = Modality.TABULAR,
        task: Union[str, DLTask] = DLTask.CLASSIFICATION,
        target_column: Optional[str] = None,
        input_column: Optional[str] = None,
        config: Optional[DLConfig] = None,
    ) -> Tuple[DLPredictor, DLLeaderboard]:
        """Execute autonomous deep learning trials and return (champion_predictor, leaderboard)."""
        resolved_modality = Modality.from_str(modality) if not isinstance(modality, Modality) else modality
        resolved_task = DLTask.from_str(task) if not isinstance(task, DLTask) else task

        cfg = config or DLConfig(
            modality=resolved_modality.value,
            task=resolved_task.value,
            mode=self.mode,
        )
        cfg.apply_mode()
        if target_column:
            cfg.data.target_column = target_column
        if input_column:
            cfg.data.input_column = input_column

        primary_metric = self.optimize_for or (
            "accuracy" if resolved_task == DLTask.CLASSIFICATION else "r2"
        )
        leaderboard = DLLeaderboard(primary_metric=primary_metric)

        # 1. Load data through modality adapter
        adapter = get_adapter(resolved_modality, options=cfg.data.options)
        raw_batch = adapter.load(data, cfg.data)

        if not raw_batch.has_targets:
            raise DLDataError(
                "No targets found in dataset. Supervised AutoDL training requires labels.",
                "Specify target column with target_column='col_name' or use folder-per-class layout.",
            )

        n_samples = raw_batch.n_samples
        if n_samples < 4:
            raise DLDataError(f"At least 4 samples are required for training, got {n_samples}.")

        # 2. Check host capability
        verdict = assess_dl_workload(resolved_modality, n_samples, cfg)
        if verdict.decision == REFUSE and not cfg.resources.assume_yes:
            raise DLTrainingError(
                f"Host capability refused this DL workload: {'; '.join(verdict.reasons)}",
                f"Remedies: {'; '.join(verdict.remedies)}",
            )

        # 3. Train/Val split
        inputs = raw_batch.inputs
        targets = np.asarray(raw_batch.targets)

        stratify = targets if (resolved_task == DLTask.CLASSIFICATION and len(np.unique(targets)) > 1) else None
        # Verify stratify class counts
        if stratify is not None:
            _, counts = np.unique(stratify, return_counts=True)
            if np.min(counts) < 2:
                stratify = None

        idx_train, idx_val = train_test_split(
            np.arange(n_samples),
            test_size=max(2, int(n_samples * cfg.data.test_size)),
            random_state=cfg.random_seed,
            stratify=stratify,
        )

        def _subset(collection: Any, indices: np.ndarray) -> Any:
            if hasattr(collection, "iloc"):
                return collection.iloc[indices]
            if isinstance(collection, (list, tuple)):
                return [collection[i] for i in indices]
            if isinstance(collection, np.ndarray):
                return collection[indices]
            return collection

        train_inputs = _subset(inputs, idx_train)
        val_inputs = _subset(inputs, idx_val)
        y_train = targets[idx_train]
        y_val = targets[idx_val]

        # 4. Modality Feature Extraction
        X_train = adapter.fit_transform(train_inputs)
        X_val = adapter.transform(val_inputs)

        # Filter out failed samples
        if adapter.failed_indices:
            keep_train = [i for i in range(len(X_train)) if i not in adapter.failed_indices]
            if keep_train:
                X_train = X_train[keep_train]
                y_train = y_train[keep_train]

        # 5. Search Candidate Trials
        candidates = self._generate_candidate_configs(cfg)
        best_trainer: Optional[DLTrainer] = None
        best_score = -float("inf")
        start_time = time.time()

        for trial_idx, candidate in enumerate(candidates, start=1):
            if (time.time() - start_time) > self.time_budget_secs:
                break

            train_cfg = DLTrainingConfig(
                epochs=cfg.training.epochs,
                learning_rate=candidate["lr"],
                hidden_sizes=candidate["hidden_sizes"],
                dropout=candidate["dropout"],
                early_stopping_patience=cfg.training.early_stopping_patience,
                seed=cfg.random_seed + trial_idx,
            )

            trainer = DLTrainer(
                task=resolved_task,
                config=train_cfg,
                batch_size=cfg.resources.batch_size,
                force_fallback=self.force_fallback,
            )

            t0 = time.perf_counter()
            status = "SUCCESS"
            error_msg = None
            try:
                trainer.fit(X_train, y_train)
                train_time_ms = (time.perf_counter() - t0) * 1000.0

                # Latency test & scoring on validation set
                t_infer = time.perf_counter()
                preds = trainer.predict(X_val)
                latency_ms = (time.perf_counter() - t_infer) * 1000.0 / max(1, len(X_val))

                if resolved_task == DLTask.CLASSIFICATION:
                    score = accuracy_score(y_val, preds)
                else:
                    score = r2_score(y_val, preds)

                composite = score * 0.85 + (1.0 / (1.0 + latency_ms * 0.05)) * 0.15
            except Exception as exc:
                status = "FAILED"
                error_msg = str(exc)
                score = 0.0
                composite = 0.0
                train_time_ms = 0.0
                latency_ms = 0.0

            arch_name = f"MLP({'-'.join(map(str, candidate['hidden_sizes']))})"
            trial = DLTrial(
                trial_id=trial_idx,
                config=cfg,
                model_name=arch_name,
                backend=trainer.backend,
                primary_metric=primary_metric,
                primary_metric_score=score,
                composite_score=composite,
                train_time_ms=train_time_ms,
                inference_latency_ms=latency_ms,
                status=status,
                error=error_msg,
                history=trainer.history,
            )
            leaderboard.add_trial(trial)

            if status == "SUCCESS" and score > best_score:
                best_score = score
                best_trainer = trainer

        # Fallback to last trainer if none succeeded
        if best_trainer is None:
            best_trainer = trainer

        champion = DLPredictor(
            adapter=adapter,
            trainer=best_trainer,
            config=cfg,
            modality=resolved_modality,
            task=resolved_task,
            input_column=raw_batch.input_column,
            target_column=raw_batch.target_column,
            metrics={"best_score": best_score, "metric": primary_metric},
        )

        return champion, leaderboard
