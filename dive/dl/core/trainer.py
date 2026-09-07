"""Modality-agnostic training core for DIVE Deep Learning - ``dive/dl/core/trainer.py``.

One trainer, two backends, five modalities. The contract that makes that work:
**every modality adapter turns its raw input into a dense ``float32`` feature
matrix**, so the trainer only ever sees ``(n_samples, n_features)`` and never
needs to know whether a row came from a CSV, a WAV file, or a video frame.

* ``torch`` present -> a real MLP trained with AdamW, cosine LR decay, gradient
  clipping, optional AMP, early stopping and in-memory best-epoch checkpointing.
* ``torch`` absent  -> scikit-learn's ``MLPClassifier``/``MLPRegressor``. Smaller
  and slower to converge, but the command still completes, which is the promise.

The torch model is assembled from stock ``nn.Sequential`` layers rather than a
locally-defined ``nn.Module`` subclass, because a class defined inside a function
is not picklable and the trained model has to survive
:func:`dive.dl.inference.predictor.save_dl_predictor`.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dive.dl.config import DLTask, DLTrainingConfig
from dive.dl.core.callbacks import CallbackList, EpochRecord, TrainingCallback
from dive.dl.core.device import DeviceInfo, select_device
from dive.dl.exceptions import DLTrainingError
from dive.utils.optional import load_optional

#: Backend identifiers reported in leaderboards and predictor metadata.
BACKEND_TORCH = "torch-mlp"
BACKEND_SKLEARN = "sklearn-mlp"


def _as_matrix(X: Any) -> np.ndarray:
    """Coerce features to a dense 2-D ``float32`` array.

    Sparse input is densified here rather than in each adapter: an MLP cannot
    consume a sparse matrix on either backend, and the adapters that produce
    sparse features (TF-IDF) already reduce dimensionality first.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    array = np.asarray(X, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise DLTrainingError(
            f"Expected a 2-D feature matrix, got shape {array.shape}.",
            "Modality adapters must flatten their features before training.",
        )
    return np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


class DLInferenceGuard(DLTrainingError):
    """Raised when predict is called before fit."""

    def __init__(self) -> None:
        super().__init__(
            "This trainer has not been fitted yet.",
            "Call fit(X, y) before predict().",
        )


class DLTrainer:
    """Fits a dense neural network on a feature matrix, on either backend."""

    def __init__(
        self,
        task: DLTask = DLTask.CLASSIFICATION,
        config: Optional[DLTrainingConfig] = None,
        device: Optional[DeviceInfo] = None,
        callbacks: Optional[Sequence[TrainingCallback]] = None,
        batch_size: int = 32,
        force_fallback: bool = False,
    ) -> None:
        self.task = DLTask.from_str(task) if not isinstance(task, DLTask) else task
        self.config = config or DLTrainingConfig()
        self.device = device or select_device("auto")
        self.callbacks = CallbackList(list(callbacks or []))
        # Batch size lives on DLResourceConfig, not DLTrainingConfig, because it
        # is a hardware property rather than an optimisation one - so it is
        # passed in rather than read off self.config.
        self.batch_size = max(2, int(batch_size))
        #: Set by ``dive dl train --no-torch`` and by the AutoDL search when it
        #: wants an explicit fallback baseline to compare against.
        self.force_fallback = bool(force_fallback)

        self.backend: str = BACKEND_SKLEARN
        self.history: List[EpochRecord] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        #: Why the torch path was not used, when it was expected to be. Surfaced
        #: in the training report so a silent downgrade is impossible.
        self.fallback_reason: Optional[str] = None

        self._model: Any = None
        self._scaler: Any = None
        self._label_encoder: Any = None
        self._target_mean: float = 0.0
        self._target_scale: float = 1.0

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def is_classification(self) -> bool:
        return self.task is DLTask.CLASSIFICATION

    @property
    def metric_name(self) -> str:
        return "accuracy" if self.is_classification else "r2"

    @property
    def uses_torch(self) -> bool:
        return self.backend == BACKEND_TORCH

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def describe(self) -> Dict[str, Any]:
        """Backend/device/architecture summary for reports and metadata."""
        return {
            "backend": self.backend,
            "device": self.device.name,
            "task": self.task.value,
            "epochs_run": len(self.history),
            "hidden_sizes": list(self.config.hidden_sizes),
            "n_features": self.n_features_,
            "n_classes": None if self.classes_ is None else int(len(self.classes_)),
            "fallback_reason": self.fallback_reason,
        }

    # ------------------------------------------------------------------
    # target handling
    # ------------------------------------------------------------------
    def _encode_targets(self, y: Any) -> np.ndarray:
        values = np.asarray(y)
        if self.is_classification:
            from sklearn.preprocessing import LabelEncoder

            # Labels arriving as a mix of int and str break LabelEncoder's sort
            # on Python 3, so they are unified to str first - the same fix the
            # NLP evaluator needed.
            self._label_encoder = LabelEncoder()
            encoded = self._label_encoder.fit_transform(values.astype(str))
            self.classes_ = np.asarray(self._label_encoder.classes_)
            if len(self.classes_) < 2:
                raise DLTrainingError(
                    f"Only one class ('{self.classes_[0]}') is present in the target.",
                    "Classification needs at least two distinct labels.",
                )
            return encoded.astype(np.int64)

        numeric = np.asarray(values, dtype=np.float64).ravel()
        # Standardising the target keeps MSE in a range where a shared default
        # learning rate works for both cent-scale and million-scale targets.
        self._target_mean = float(np.mean(numeric))
        self._target_scale = float(np.std(numeric)) or 1.0
        return ((numeric - self._target_mean) / self._target_scale).astype(np.float32)

    def _decode_predictions(self, raw: np.ndarray) -> np.ndarray:
        if self.is_classification:
            if self._label_encoder is None:
                return raw
            return np.asarray(self._label_encoder.inverse_transform(raw.astype(int)))
        return raw * self._target_scale + self._target_mean

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            if self.is_classification:
                from sklearn.metrics import accuracy_score

                return float(accuracy_score(y_true, y_pred))
            from sklearn.metrics import r2_score

            return float(r2_score(y_true, y_pred))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
        time_budget_secs: Optional[float] = None,
    ) -> "DLTrainer":
        """Fit on ``X``/``y``, validating on ``X_val`` or an internal holdout."""
        features = _as_matrix(X)
        targets = self._encode_targets(y)
        if len(features) != len(targets):
            raise DLTrainingError(
                f"Feature/target length mismatch: {len(features)} rows vs {len(targets)} targets."
            )
        if len(features) < 4:
            raise DLTrainingError(
                f"Only {len(features)} usable row(s); deep learning needs more.",
                "Provide at least a few dozen labelled examples.",
            )
        self.n_features_ = int(features.shape[1])

        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        features = self._scaler.fit_transform(features).astype(np.float32)

        if X_val is not None and y_val is not None:
            validation: Optional[Tuple[np.ndarray, np.ndarray]] = (
                self._scaler.transform(_as_matrix(X_val)).astype(np.float32),
                self._encode_validation_targets(y_val),
            )
            train_features, train_targets = features, targets
        else:
            train_features, train_targets, validation = self._internal_split(features, targets)

        torch = None if self.force_fallback else load_optional("torch")
        if torch is not None:
            try:
                self._fit_torch(torch, train_features, train_targets, validation, time_budget_secs)
                self.backend = BACKEND_TORCH
                return self
            except DLTrainingError:
                raise
            except Exception as exc:
                # A torch failure must not lose the run: fall through to sklearn
                # and record why, rather than surfacing a stack trace.
                self.history = []
                self.fallback_reason = (
                    f"the PyTorch path failed ({type(exc).__name__}: {exc}); "
                    "trained with the scikit-learn fallback instead"
                )

        if torch is None and not self.force_fallback and self.fallback_reason is None:
            self.fallback_reason = (
                "torch is not installed; trained with the scikit-learn fallback"
            )
        self._fit_sklearn(train_features, train_targets, validation)
        self.backend = BACKEND_SKLEARN
        return self

    def _encode_validation_targets(self, y_val: Any) -> np.ndarray:
        values = np.asarray(y_val)
        if self.is_classification:
            if self._label_encoder is None:
                return values
            known = set(str(label) for label in self._label_encoder.classes_)
            as_text = values.astype(str)
            # A validation label unseen in training cannot be encoded; drop the
            # comparison rather than crash, and let the caller see a lower score.
            mapped = np.array(
                [
                    self._label_encoder.transform([label])[0] if label in known else -1
                    for label in as_text
                ],
                dtype=np.int64,
            )
            return mapped
        numeric = np.asarray(values, dtype=np.float64).ravel()
        return ((numeric - self._target_mean) / self._target_scale).astype(np.float32)

    def _internal_split(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Carve a validation holdout out of the training data.

        Returns ``None`` for the holdout on datasets too small to spare rows, in
        which case early stopping falls back to watching the training loss.
        """
        from sklearn.model_selection import train_test_split

        if len(features) < 20:
            return features, targets, None
        stratify = None
        if self.is_classification:
            _, counts = np.unique(targets, return_counts=True)
            if counts.min() >= 2:
                stratify = targets
        try:
            a, b, c, d = train_test_split(
                features,
                targets,
                test_size=0.2,
                random_state=self.config.seed,
                stratify=stratify,
            )
            return a, c, (b, d)
        except Exception:
            return features, targets, None

    # ------------------------------------------------------------------
    # torch backend
    # ------------------------------------------------------------------
    def _build_torch_model(self, torch: Any, n_features: int, n_outputs: int) -> Any:
        """Assemble the MLP from stock layers so the result stays picklable."""
        nn = torch.nn
        layers: List[Any] = []
        width = n_features
        for size in self.config.hidden_sizes:
            layers.append(nn.Linear(width, int(size)))
            layers.append(nn.BatchNorm1d(int(size)))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(float(self.config.dropout)))
            width = int(size)
        layers.append(nn.Linear(width, n_outputs))
        return nn.Sequential(*layers)

    def _fit_torch(
        self,
        torch: Any,
        features: np.ndarray,
        targets: np.ndarray,
        validation: Optional[Tuple[np.ndarray, np.ndarray]],
        time_budget_secs: Optional[float],
    ) -> None:
        torch.manual_seed(int(self.config.seed))
        nn = torch.nn
        device = torch.device(self.device.name)

        n_outputs = int(len(self.classes_)) if self.is_classification else 1
        model = self._build_torch_model(torch, features.shape[1], n_outputs).to(device)

        criterion = nn.CrossEntropyLoss() if self.is_classification else nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        epochs = max(1, int(self.config.epochs))
        try:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        except Exception:
            scheduler = None

        # AMP is a CUDA-only win and its constructor moved between torch
        # versions, so it is probed rather than assumed.
        scaler = None
        if self.config.mixed_precision and self.device.name == "cuda":
            for factory in (
                lambda: torch.amp.GradScaler("cuda"),
                lambda: torch.cuda.amp.GradScaler(),
            ):
                try:
                    scaler = factory()
                    break
                except Exception:
                    scaler = None

        x_train = torch.from_numpy(features)
        y_train = torch.from_numpy(
            targets if self.is_classification else targets.reshape(-1, 1)
        )
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        # BatchNorm1d rejects a batch of one, so a trailing single-row batch is
        # dropped whenever that would not empty the loader.
        batch_size = max(2, min(self.batch_size, len(features)))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(features) % batch_size == 1 and len(features) > batch_size,
        )

        val_tensors = None
        if validation is not None:
            val_x = torch.from_numpy(validation[0]).to(device)
            val_y_raw = validation[1]
            val_tensors = (val_x, val_y_raw)

        best_state = copy.deepcopy(model.state_dict())
        best_metric = -np.inf
        stale = 0
        started = time.perf_counter()
        self.callbacks.on_train_begin(
            epochs,
            {"backend": BACKEND_TORCH, "device": self.device.name, "features": features.shape[1]},
        )

        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            running = 0.0
            seen = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(batch_x)
                        loss = criterion(output, batch_y)
                    scaler.scale(loss).backward()
                    if self.config.gradient_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), float(self.config.gradient_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    if self.config.gradient_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), float(self.config.gradient_clip))
                    optimizer.step()
                running += float(loss.detach().cpu()) * len(batch_x)
                seen += len(batch_x)
            if scheduler is not None:
                scheduler.step()

            train_loss = running / max(seen, 1)
            val_loss: Optional[float] = None
            val_metric: Optional[float] = None
            if val_tensors is not None:
                model.eval()
                with torch.no_grad():
                    logits = model(val_tensors[0])
                    if self.is_classification:
                        truth = np.asarray(val_tensors[1])
                        predicted = logits.argmax(dim=1).cpu().numpy()
                        mask = truth >= 0
                        val_metric = (
                            float((predicted[mask] == truth[mask]).mean()) if mask.any() else 0.0
                        )
                        keep = torch.from_numpy(np.flatnonzero(mask)).to(device)
                        if len(keep):
                            val_loss = float(
                                criterion(
                                    logits.index_select(0, keep),
                                    torch.from_numpy(truth[mask]).to(device),
                                ).cpu()
                            )
                    else:
                        truth = torch.from_numpy(
                            np.asarray(val_tensors[1], dtype=np.float32).reshape(-1, 1)
                        ).to(device)
                        val_loss = float(criterion(logits, truth).cpu())
                        val_metric = self._score(
                            np.asarray(val_tensors[1]).ravel(),
                            logits.cpu().numpy().ravel(),
                        )

            watched = val_metric if val_metric is not None else -train_loss
            improved = watched > best_metric
            if improved:
                best_metric = watched
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metric=val_metric,
                metric_name=self.metric_name,
                seconds=time.perf_counter() - epoch_started,
                improved=improved,
            )
            self.history.append(record)
            self.callbacks.on_epoch_end(record)

            if self.config.early_stopping_patience and stale >= self.config.early_stopping_patience:
                break
            if time_budget_secs and (time.perf_counter() - started) >= float(time_budget_secs):
                break

        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._torch_module = torch
        self.callbacks.on_train_end(self.history)

    # ------------------------------------------------------------------
    # sklearn fallback
    # ------------------------------------------------------------------
    def _fit_sklearn(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        hidden = tuple(int(size) for size in self.config.hidden_sizes) or (128,)
        common = {
            "hidden_layer_sizes": hidden,
            "learning_rate_init": float(self.config.learning_rate),
            "alpha": max(float(self.config.weight_decay), 1e-4),
            # sklearn's MLP has no epoch callback, so its own early stopping is
            # used and `max_iter` carries the epoch budget.
            "max_iter": max(50, int(self.config.epochs) * 25),
            "early_stopping": len(features) >= 50,
            "n_iter_no_change": max(2, int(self.config.early_stopping_patience)),
            "random_state": int(self.config.seed),
        }
        model = MLPClassifier(**common) if self.is_classification else MLPRegressor(**common)

        self.callbacks.on_train_begin(
            1, {"backend": BACKEND_SKLEARN, "device": "cpu", "features": features.shape[1]}
        )
        started = time.perf_counter()
        import warnings

        with warnings.catch_warnings():
            # A convergence warning here is expected on small data and is
            # reported through the epoch record instead.
            warnings.simplefilter("ignore")
            model.fit(features, targets)
        elapsed = time.perf_counter() - started

        val_metric = None
        if validation is not None:
            truth = np.asarray(validation[1])
            predicted = model.predict(validation[0])
            if self.is_classification:
                mask = truth >= 0
                val_metric = float((predicted[mask] == truth[mask]).mean()) if mask.any() else 0.0
            else:
                val_metric = self._score(truth.ravel(), np.asarray(predicted).ravel())

        losses = getattr(model, "loss_curve_", None)
        final_loss = float(losses[-1]) if losses else float(getattr(model, "loss_", 0.0) or 0.0)
        record = EpochRecord(
            epoch=1,
            train_loss=final_loss,
            val_metric=val_metric,
            metric_name=self.metric_name,
            seconds=elapsed,
            improved=True,
        )
        self.history = [record]
        self.callbacks.on_epoch_end(record)
        self._model = model
        self.callbacks.on_train_end(self.history)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _prepare(self, X: Any) -> np.ndarray:
        if not self.fitted:
            raise DLInferenceGuard()
        matrix = _as_matrix(X)
        if self.n_features_ is not None and matrix.shape[1] != self.n_features_:
            raise DLTrainingError(
                f"Model expects {self.n_features_} features but received {matrix.shape[1]}.",
                "The same modality adapter must featurise training and inference input.",
            )
        return self._scaler.transform(matrix).astype(np.float32)

    def predict(self, X: Any) -> np.ndarray:
        prepared = self._prepare(X)
        if self.uses_torch:
            torch = self._torch_module
            with torch.no_grad():
                logits = self._model(torch.from_numpy(prepared).to(torch.device(self.device.name)))
                if self.is_classification:
                    raw = logits.argmax(dim=1).cpu().numpy()
                else:
                    raw = logits.cpu().numpy().ravel()
        else:
            raw = np.asarray(self._model.predict(prepared))
        return self._decode_predictions(raw)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_classification:
            raise DLTrainingError(
                "predict_proba is only available for classification models.",
                f"This trainer was built for '{self.task.value}'.",
            )
        prepared = self._prepare(X)
        if self.uses_torch:
            torch = self._torch_module
            with torch.no_grad():
                logits = self._model(torch.from_numpy(prepared).to(torch.device(self.device.name)))
                return torch.softmax(logits, dim=1).cpu().numpy()
        return np.asarray(self._model.predict_proba(prepared))

    # Pickling: the torch module itself pickles, but the *module object* cached
    # on the instance does not, so it is dropped and re-imported on load.
    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_torch_module", None)
        state["callbacks"] = CallbackList([])
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.backend == BACKEND_TORCH:
            torch = load_optional("torch")
            if torch is None:
                raise DLTrainingError(
                    "This model was trained with PyTorch, which is not installed here.",
                    "Install it with 'pip install torch', or retrain with --no-torch.",
                )
            self._torch_module = torch
