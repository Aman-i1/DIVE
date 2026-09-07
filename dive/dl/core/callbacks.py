"""Training callbacks for DIVE Deep Learning - ``dive/dl/core/callbacks.py``.

Progress is routed through :class:`~dive.utils.logging.Console` using the same
``step`` / ``model_result`` primitives the tabular ML trainer uses, so an epoch
line in ``dive dl train`` looks like a model line in ``dive ml train``. Nothing
here prints a raw glyph; the console's symbol table is the only source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dive.utils.logging import Console, Style, get_console


@dataclass
class EpochRecord:
    """One row of training history."""

    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    val_metric: Optional[float] = None
    metric_name: str = "score"
    seconds: float = 0.0
    #: Set by the trainer when this epoch beat every previous one.
    improved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": round(float(self.train_loss), 6),
            "val_loss": None if self.val_loss is None else round(float(self.val_loss), 6),
            "val_metric": None if self.val_metric is None else round(float(self.val_metric), 6),
            "metric_name": self.metric_name,
            "seconds": round(float(self.seconds), 3),
            "improved": self.improved,
        }


class TrainingCallback:
    """No-op base class. Override only the hooks you need."""

    def on_train_begin(self, total_epochs: int, context: Optional[Dict[str, Any]] = None) -> None:
        """Called once before the first epoch."""

    def on_epoch_end(self, record: EpochRecord) -> None:
        """Called after every epoch, including the one that triggers early stop."""

    def on_train_end(self, history: List[EpochRecord]) -> None:
        """Called once after the loop finishes or is stopped early."""


class ConsoleCallback(TrainingCallback):
    """Report progress to the terminal in the platform's grammar."""

    def __init__(self, console: Optional[Console] = None, label: str = "training") -> None:
        self.console = console if console is not None else get_console()
        self.label = label
        self._total = 0

    def on_train_begin(self, total_epochs: int, context: Optional[Dict[str, Any]] = None) -> None:
        self._total = max(1, int(total_epochs))
        details = context or {}
        summary = ", ".join(f"{key}={value}" for key, value in details.items())
        message = f"Training {self.label} for {self._total} epoch(s)"
        self.console.info(f"  {message}{f' ({summary})' if summary else ''}")

    def on_epoch_end(self, record: EpochRecord) -> None:
        parts = [f"loss={record.train_loss:.4f}"]
        if record.val_loss is not None:
            parts.append(f"val_loss={record.val_loss:.4f}")
        if record.val_metric is not None:
            parts.append(f"{record.metric_name}={record.val_metric:.4f}")
        marker = self.console.status_symbol("up") if record.improved else " "
        self.console.step(
            record.epoch,
            self._total,
            f"{'  '.join(parts)}  "
            f"{self.console.paint(f'({record.seconds:.1f}s)', Style.MUTED)} {marker}",
        )

    def on_train_end(self, history: List[EpochRecord]) -> None:
        if not history:
            return
        best = max(
            history,
            key=lambda record: (
                record.val_metric if record.val_metric is not None else -record.train_loss
            ),
        )
        detail = (
            f"{best.metric_name}={best.val_metric:.4f}"
            if best.val_metric is not None
            else f"loss={best.train_loss:.4f}"
        )
        self.console.info(
            f"  {self.console.status_symbol('ok')} "
            f"best epoch {best.epoch}/{len(history)}  {detail}"
        )


@dataclass
class HistoryCallback(TrainingCallback):
    """Collect every :class:`EpochRecord` for later serialisation."""

    records: List[EpochRecord] = field(default_factory=list)

    def on_epoch_end(self, record: EpochRecord) -> None:
        self.records.append(record)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.records]


class CallbackList(TrainingCallback):
    """Fan one hook out to many callbacks.

    A raising callback must not take the training run down with it - progress
    reporting is not part of the result - so each dispatch is guarded.
    """

    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None) -> None:
        self.callbacks: List[TrainingCallback] = list(callbacks or [])

    def add(self, callback: TrainingCallback) -> "CallbackList":
        self.callbacks.append(callback)
        return self

    def _dispatch(self, hook: str, *args: Any) -> None:
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if not callable(method):
                continue
            try:
                method(*args)
            except Exception:
                pass

    def on_train_begin(self, total_epochs: int, context: Optional[Dict[str, Any]] = None) -> None:
        self._dispatch("on_train_begin", total_epochs, context)

    def on_epoch_end(self, record: EpochRecord) -> None:
        self._dispatch("on_epoch_end", record)

    def on_train_end(self, history: List[EpochRecord]) -> None:
        self._dispatch("on_train_end", history)
