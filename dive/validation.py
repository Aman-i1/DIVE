"""The crosscheck engine: "is my data ready?" answered before training starts.

Every check returns a :class:`CheckResult` with a PASS / WARN / FAIL verdict, and
the whole set is bundled into a :class:`ValidationReport`. The same report type is
produced by ``dive validate`` (no training) and embedded inside ``dive train``,
so the two can never drift apart.

The checks, and what a failure actually means:

``target_health``
    Class imbalance, a near-constant target, or heavy target missingness. These
    come from :class:`dive.data_intelligence.DataIntelligence`, surfaced here as
    verdicts rather than left sitting in the profile unused.
``target_leakage``
    A single feature almost perfectly predicts the target. Usually an outcome
    column, a post-hoc identifier, or a re-encoding of the label.
``duplicate_rows``
    Identical rows appear in both the train and holdout splits, so the holdout
    score is partly a memory test.
``train_holdout_drift``
    A numeric feature is distributed differently across the split. Expected for
    time-series splits; suspicious for random ones.
``missing_data``
    Columns missing enough values that imputation is doing the modelling.
``cv_stability``
    Per-fold score variance for the winning model - a high spread means the
    single headline number is not reproducible.
``predict_schema``
    Incoming columns/dtypes vs. what the model was trained on. This one hard-fails
    at predict time instead of silently misaligning columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

_SEVERITY = {PASS: 0, SKIP: 1, WARN: 2, FAIL: 3}

# A feature this predictive of the target is treated as leakage, not signal.
LEAKAGE_THRESHOLD = 0.98
# Population Stability Index above this is a meaningful distribution shift.
PSI_THRESHOLD = 0.2
# Kolmogorov-Smirnov p-value below this rejects "same distribution".
KS_ALPHA = 0.01
# Fold-score spread above this fraction of the mean is unstable.
CV_INSTABILITY_RATIO = 0.10
# Columns missing more than this fraction are called out.
HIGH_MISSING_PCT = 40.0


@dataclass
class CheckResult:
    """The outcome of a single crosscheck."""

    name: str
    status: str
    summary: str
    details: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return self.status == FAIL

    @property
    def is_warning(self) -> bool:
        return self.status == WARN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": list(self.details),
            "metrics": dict(self.metrics),
        }


@dataclass
class ValidationReport:
    """A collection of check results plus the context they were computed in."""

    checks: List[CheckResult] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: Optional[CheckResult]) -> None:
        if result is not None:
            self.checks.append(result)

    def get(self, name: str) -> Optional[CheckResult]:
        for check in self.checks:
            if check.name == name:
                return check
        return None

    @property
    def has_failures(self) -> bool:
        return any(check.is_failure for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(check.is_warning for check in self.checks)

    @property
    def worst_status(self) -> str:
        if not self.checks:
            return PASS
        return max((c.status for c in self.checks), key=lambda s: _SEVERITY.get(s, 0))

    def counts(self) -> Dict[str, int]:
        tally = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for check in self.checks:
            tally[check.status] = tally.get(check.status, 0) + 1
        return tally

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worst_status": self.worst_status,
            "counts": self.counts(),
            "context": dict(self.context),
            "checks": [check.to_dict() for check in self.checks],
        }

    def render(self, console: Any = None) -> str:
        """Return a printable pass/warn/fail report.

        When ``console`` is supplied and it is emitting colour, each verdict mark
        is tinted. The escape wraps the whole ``[PASS]`` token, leaving the inner
        word intact for anything that greps the output. ``check.status`` and
        ``check.summary`` themselves are never coloured, so ``to_dict`` and the
        HTML report stay free of escape sequences.
        """
        marks = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}
        tints = {}
        if console is not None and getattr(console, "color", False):
            from dive.utils.logging import Style

            tints = {
                PASS: (Style.SUCCESS,),
                WARN: (Style.WARN,),
                FAIL: (Style.ERROR, Style.BOLD),
                SKIP: (Style.MUTED,),
            }

        def paint(text: str, status: str) -> str:
            styles = tints.get(status)
            if not styles:
                return text
            return console.paint(text, *styles)

        lines: List[str] = []
        for check in self.checks:
            mark = paint(marks.get(check.status, "[????]"), check.status)
            lines.append(f" {mark} {check.name}: {check.summary}")
            for detail in check.details[:8]:
                lines.append(f"          {detail}")
            if len(check.details) > 8:
                lines.append(f"          ... and {len(check.details) - 8} more")
        tally = self.counts()
        lines.append("")
        summary = (
            f" Summary: {tally[PASS]} passed, {tally[WARN]} warning(s), "
            f"{tally[FAIL]} failure(s)"
            + (f", {tally[SKIP]} skipped" if tally.get(SKIP) else "")
        )
        lines.append(paint(summary, self.worst_status))
        return "\n".join(lines)

# ----------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------
def check_target_health(profile: Dict[str, Any]) -> CheckResult:
    """Surface imbalance / near-constant / missing-target findings as verdicts."""
    details: List[str] = []
    status = PASS
    problem_type = profile.get("problem_type")

    if profile.get("target_near_constant"):
        status = FAIL
        if problem_type == "classification":
            details.append(
                "One class covers >=99% of rows - a model that always predicts "
                "that class would look accurate while learning nothing."
            )
        else:
            details.append(
                "The target has almost no variance, so there is effectively "
                "nothing to predict."
            )

    missing_pct = float(profile.get("target_missing_pct") or 0.0)
    if missing_pct > 0:
        message = f"{missing_pct:.1f}% of target values are missing (those rows are dropped)."
        if missing_pct > 20:
            status = max(status, WARN, key=lambda s: _SEVERITY[s])
        details.append(message)

    ratio = profile.get("imbalance_ratio")
    if problem_type == "classification" and ratio:
        minority = profile.get("minority_class_count")
        if ratio > 10:
            status = max(status, WARN, key=lambda s: _SEVERITY[s])
            details.append(
                f"Severe class imbalance {ratio:.1f}:1 (smallest class has "
                f"{minority} rows). Accuracy will be misleading - read the "
                "balanced-accuracy and F1 columns instead."
            )
        elif profile.get("is_imbalanced"):
            details.append(
                f"Class imbalance {ratio:.1f}:1 - class_weight='balanced' is "
                "applied automatically where the model supports it."
            )
        if minority is not None and minority < 5:
            status = FAIL
            details.append(
                f"The smallest class has only {minority} row(s), which is too few "
                "to appear in every cross-validation fold."
            )

    summary = {
        PASS: "target looks usable",
        WARN: "target is usable but skewed",
        FAIL: "target is not usable as-is",
    }[status]
    return CheckResult(
        name="target_health",
        status=status,
        summary=summary,
        details=details,
        metrics={
            "problem_type": problem_type,
            "imbalance_ratio": ratio,
            "target_missing_pct": missing_pct,
            "n_classes": profile.get("n_classes"),
            "minority_class_count": profile.get("minority_class_count"),
        },
    )


def check_target_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    threshold: float = LEAKAGE_THRESHOLD,
    max_features: int = 400,
) -> CheckResult:
    """Flag features that predict the target almost perfectly.

    Numeric features are scored by \\|Pearson r\\| against a numeric target, and by
    normalised mutual information otherwise. Categorical features are scored by
    normalised mutual information. A score at or above ``threshold`` is reported
    as leakage: legitimate predictors essentially never reach 0.98.
    """
    if X.shape[1] == 0:
        return CheckResult("target_leakage", SKIP, "no features to check")

    columns = list(X.columns)[:max_features]
    truncated = len(X.columns) - len(columns)
    scores: Dict[str, float] = {}

    y_clean = y.reset_index(drop=True)
    y_numeric = pd.to_numeric(y_clean, errors="coerce")
    y_is_numeric = bool(y_numeric.notna().all())

    for column in columns:
        series = X[column].reset_index(drop=True)
        try:
            if pd.api.types.is_numeric_dtype(series):
                score = _numeric_association(series, y_clean, y_numeric, y_is_numeric, problem_type)
            else:
                score = _categorical_association(series, y_clean, problem_type)
        except Exception:
            continue
        if score is not None and np.isfinite(score):
            scores[str(column)] = float(score)

    leaking = {name: value for name, value in scores.items() if value >= threshold}
    suspicious = {
        name: value
        for name, value in scores.items()
        if 0.90 <= value < threshold and name not in leaking
    }

    details: List[str] = []
    if leaking:
        status = FAIL
        summary = f"{len(leaking)} feature(s) almost perfectly predict the target"
        for name, value in sorted(leaking.items(), key=lambda kv: -kv[1]):
            details.append(f"- {name}: association {value:.4f} (>= {threshold})")
        details.append(
            "This is the signature of target leakage: the column probably encodes "
            "the answer. Drop it, or the reported scores will not survive contact "
            "with real data."
        )
    elif suspicious:
        status = WARN
        summary = f"{len(suspicious)} feature(s) are unusually predictive"
        for name, value in sorted(suspicious.items(), key=lambda kv: -kv[1]):
            details.append(f"- {name}: association {value:.4f}")
        details.append("Strong but below the leakage threshold - worth a sanity check.")
    else:
        status = PASS
        summary = "no single feature dominates the target"

    if truncated > 0:
        details.append(f"Note: only the first {len(columns)} of {X.shape[1]} columns were scored.")

    return CheckResult(
        name="target_leakage",
        status=status,
        summary=summary,
        details=details,
        metrics={
            "threshold": threshold,
            "leaking_features": leaking,
            "suspicious_features": suspicious,
            "n_scored": len(scores),
            "n_truncated": max(0, truncated),
        },
    )


def _numeric_association(
    series: pd.Series,
    y_raw: pd.Series,
    y_numeric: pd.Series,
    y_is_numeric: bool,
    problem_type: str,
) -> Optional[float]:
    """Association between a numeric feature and the target, in [0, 1]."""
    mask = series.notna()
    if y_is_numeric:
        mask &= y_numeric.notna()
    if mask.sum() < 5:
        return None
    values = series[mask]
    if values.nunique() <= 1:
        return 0.0

    if problem_type == "regression" and y_is_numeric:
        correlation = np.corrcoef(values.astype(float), y_numeric[mask].astype(float))[0, 1]
        return float(abs(correlation)) if np.isfinite(correlation) else None

    # Classification: a numeric feature that separates classes perfectly shows up
    # as near-1.0 normalised mutual information once binned.
    codes = pd.factorize(y_raw[mask])[0]
    binned = _bin_numeric(values)
    return _normalised_mutual_info(binned, codes)


def _categorical_association(
    series: pd.Series, y_raw: pd.Series, problem_type: str
) -> Optional[float]:
    """Normalised mutual information between a categorical feature and the target."""
    mask = series.notna() & y_raw.notna()
    if mask.sum() < 5:
        return None
    feature_codes = pd.factorize(series[mask].astype(str))[0]
    if len(np.unique(feature_codes)) <= 1:
        return 0.0
    # A column with one distinct value per row (an ID) trivially "predicts"
    # everything; that is an ID problem, not leakage, and is reported elsewhere.
    if len(np.unique(feature_codes)) == mask.sum():
        return 0.0
    if problem_type == "regression":
        target_codes = _bin_numeric(pd.to_numeric(y_raw[mask], errors="coerce").fillna(0))
    else:
        target_codes = pd.factorize(y_raw[mask].astype(str))[0]
    return _normalised_mutual_info(feature_codes, target_codes)


def _bin_numeric(values: pd.Series, bins: int = 12) -> np.ndarray:
    """Discretise a numeric series into integer codes for MI computation.

    Low-cardinality columns are used as-is: quantile binning a 0/1 indicator
    collapses every row into one bin (all quantiles are equal, and
    ``duplicates="drop"`` removes the edges), which would report zero mutual
    information for a column that perfectly encodes the target.
    """
    values = values.astype(float)
    distinct = values.nunique(dropna=True)
    if distinct <= bins:
        return pd.factorize(values)[0]
    try:
        binned = pd.qcut(values, q=bins, duplicates="drop", labels=False)
    except (ValueError, TypeError):
        try:
            binned = pd.cut(values, bins=bins, labels=False)
        except (ValueError, TypeError):
            return pd.factorize(values.astype(str))[0]
    codes = pd.Series(binned).fillna(-1).to_numpy()
    if len(np.unique(codes)) <= 1:
        return pd.factorize(values)[0]
    return codes


def _normalised_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information scaled into [0, 1] by the smaller marginal entropy."""
    from sklearn.metrics import mutual_info_score

    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    mutual = float(mutual_info_score(a, b))
    entropy_a, entropy_b = _entropy(a), _entropy(b)
    denominator = min(entropy_a, entropy_b)
    if denominator <= 1e-12:
        return 0.0
    return float(min(1.0, mutual / denominator))


def _entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum())
def check_duplicate_rows(
    X: pd.DataFrame,
    train_index: Optional[Sequence] = None,
    test_index: Optional[Sequence] = None,
) -> CheckResult:
    """Detect duplicated feature rows, especially ones spanning the split.

    A row present in both train and holdout turns the holdout score into a
    partial memory test - the model has already seen the answer.
    """
    if X.shape[0] < 2:
        return CheckResult("duplicate_rows", SKIP, "not enough rows to compare")

    try:
        hashed = pd.util.hash_pandas_object(X.astype(str), index=False)
    except Exception:
        return CheckResult("duplicate_rows", SKIP, "rows could not be hashed for comparison")

    total_duplicates = int(hashed.duplicated().sum())
    details: List[str] = []
    metrics: Dict[str, Any] = {
        "duplicate_rows": total_duplicates,
        "duplicate_pct": round(100.0 * total_duplicates / len(X), 3),
    }
    status = PASS
    summary = "no duplicate rows"

    if total_duplicates:
        pct = metrics["duplicate_pct"]
        status = WARN
        summary = f"{total_duplicates} duplicate row(s) ({pct:.2f}% of the data)"
        details.append(
            "Duplicated feature rows inflate holdout scores and bias any "
            "frequency-based encoding."
        )

    if train_index is not None and test_index is not None:
        hashes = pd.Series(hashed.to_numpy(), index=X.index)
        train_hashes = set(hashes.loc[list(train_index)].tolist())
        test_hashes = hashes.loc[list(test_index)]
        crossing = int(test_hashes.isin(train_hashes).sum())
        metrics["rows_in_both_splits"] = crossing
        metrics["holdout_rows"] = int(len(test_hashes))
        if crossing:
            leak_pct = 100.0 * crossing / max(len(test_hashes), 1)
            metrics["holdout_contamination_pct"] = round(leak_pct, 3)
            status = FAIL if leak_pct >= 1.0 else WARN
            summary = (
                f"{crossing} holdout row(s) ({leak_pct:.2f}%) also appear in the "
                "training split"
            )
            details.append(
                "This is duplicate-row leakage: those holdout rows were memorised, "
                "not predicted. Deduplicate before training, or group-split so "
                "identical records stay on one side."
            )

    return CheckResult(
        name="duplicate_rows",
        status=status,
        summary=summary,
        details=details,
        metrics=metrics,
    )


def check_train_holdout_drift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_features: int = 200,
) -> CheckResult:
    """Compare numeric feature distributions across the split (KS test + PSI)."""
    numeric_columns = [
        column
        for column in X_train.select_dtypes(include=np.number).columns
        if column in X_test.columns
    ][:max_features]

    if not numeric_columns:
        return CheckResult("train_holdout_drift", SKIP, "no numeric features to compare")
    if len(X_test) < 20 or len(X_train) < 20:
        return CheckResult(
            "train_holdout_drift", SKIP, "too few rows for a meaningful comparison"
        )

    try:
        from scipy.stats import ks_2samp

        have_scipy = True
    except Exception:
        have_scipy = False

    drifted: Dict[str, Dict[str, float]] = {}
    for column in numeric_columns:
        train_values = pd.to_numeric(X_train[column], errors="coerce").dropna()
        test_values = pd.to_numeric(X_test[column], errors="coerce").dropna()
        if len(train_values) < 10 or len(test_values) < 10:
            continue

        psi = _population_stability_index(train_values, test_values)
        entry: Dict[str, float] = {"psi": round(float(psi), 4)}
        flagged = psi >= PSI_THRESHOLD

        if have_scipy:
            try:
                statistic, p_value = ks_2samp(train_values, test_values)
                entry["ks_statistic"] = round(float(statistic), 4)
                entry["ks_p_value"] = float(p_value)
                # Require both signals: KS alone flags trivial shifts on big data.
                flagged = flagged and p_value < KS_ALPHA
            except Exception:
                pass

        if flagged:
            drifted[str(column)] = entry

    details: List[str] = []
    if drifted:
        status = WARN
        summary = f"{len(drifted)} numeric feature(s) differ across the split"
        for name, entry in sorted(
            drifted.items(), key=lambda kv: -kv[1]["psi"]
        )[:10]:
            ks = (
                f", KS p={entry['ks_p_value']:.2e}" if "ks_p_value" in entry else ""
            )
            details.append(f"- {name}: PSI={entry['psi']:.3f}{ks}")
        details.append(
            "Expected when splitting chronologically (--time-series). For a random "
            "split it suggests ordered or grouped data, so the holdout score may "
            "not describe future rows."
        )
    else:
        status = PASS
        summary = "train and holdout distributions agree"

    if not have_scipy:
        details.append("scipy is not installed - PSI was used without the KS test.")

    return CheckResult(
        name="train_holdout_drift",
        status=status,
        summary=summary,
        details=details,
        metrics={
            "drifted_features": drifted,
            "n_compared": len(numeric_columns),
            "psi_threshold": PSI_THRESHOLD,
        },
    )


def _population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10
) -> float:
    """PSI between two numeric samples, using quantile bins from ``expected``."""
    try:
        quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(quantiles) < 3:
            return 0.0
        quantiles[0], quantiles[-1] = -np.inf, np.inf
        expected_counts = np.histogram(expected, bins=quantiles)[0].astype(float)
        actual_counts = np.histogram(actual, bins=quantiles)[0].astype(float)
        expected_share = expected_counts / max(expected_counts.sum(), 1)
        actual_share = actual_counts / max(actual_counts.sum(), 1)
        # Floor empty bins so the log stays finite.
        floor = 1e-6
        expected_share = np.clip(expected_share, floor, None)
        actual_share = np.clip(actual_share, floor, None)
        return float(
            np.sum((actual_share - expected_share) * np.log(actual_share / expected_share))
        )
    except Exception:
        return 0.0


def check_missing_data(profile: Dict[str, Any]) -> CheckResult:
    """Report columns whose missingness makes them mostly imputation artefacts."""
    missing_pct: Dict[str, float] = profile.get("missing_pct", {}) or {}
    if not missing_pct:
        return CheckResult("missing_data", PASS, "no missing values")

    heavy = {
        column: pct for column, pct in missing_pct.items() if pct >= HIGH_MISSING_PCT
    }
    empty = {column: pct for column, pct in missing_pct.items() if pct >= 99.0}
    overall = float(profile.get("total_missing_pct") or 0.0)

    details: List[str] = []
    if empty:
        status = FAIL
        summary = f"{len(empty)} column(s) are effectively empty"
        for column, pct in sorted(empty.items(), key=lambda kv: -kv[1])[:10]:
            details.append(f"- {column}: {pct:.1f}% missing")
        details.append("These columns carry no information and should be dropped.")
    elif heavy:
        status = WARN
        summary = f"{len(heavy)} column(s) are more than {HIGH_MISSING_PCT:.0f}% missing"
        for column, pct in sorted(heavy.items(), key=lambda kv: -kv[1])[:10]:
            details.append(f"- {column}: {pct:.1f}% missing")
        details.append(
            "Median/most-frequent imputation will supply most of these values, so "
            "the model is partly learning from filled-in data."
        )
    else:
        status = PASS
        summary = f"missing values are manageable ({overall:.2f}% of cells)"

    return CheckResult(
        name="missing_data",
        status=status,
        summary=summary,
        details=details,
        metrics={
            "total_missing_pct": round(overall, 3),
            "high_missing_columns": {k: round(v, 2) for k, v in heavy.items()},
        },
    )


def check_cv_stability(
    fold_scores: Sequence[float],
    model_name: str = "best model",
    scoring: str = "score",
) -> CheckResult:
    """Report per-fold variance for the winning model, not just the mean."""
    scores = [float(s) for s in (fold_scores or []) if s is not None and np.isfinite(s)]
    if len(scores) < 2:
        return CheckResult(
            "cv_stability",
            SKIP,
            "no per-fold scores available for the best model",
            details=[
                "Boosted models and stacked ensembles are scored on the holdout "
                "instead of by cross-validation, so there are no fold scores."
            ],
        )

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    spread = float(max(scores) - min(scores))
    relative = std / max(abs(mean), 1e-9)

    details = [
        f"Fold scores: {', '.join(f'{s:.4f}' for s in scores)}",
        f"mean={mean:.4f}  std={std:.4f}  range={spread:.4f}",
    ]

    if relative >= CV_INSTABILITY_RATIO:
        status = WARN
        summary = (
            f"{model_name} is unstable across folds "
            f"(std is {relative * 100:.1f}% of the mean {scoring})"
        )
        details.append(
            "The headline score depends heavily on which rows landed in which "
            "fold. Expect real-world performance anywhere in the fold range, and "
            "prefer more data or a simpler model."
        )
    else:
        status = PASS
        summary = f"{model_name} scores consistently across folds (std={std:.4f})"

    return CheckResult(
        name="cv_stability",
        status=status,
        summary=summary,
        details=details,
        metrics={
            "fold_scores": scores,
            "mean": mean,
            "std": std,
            "range": spread,
            "relative_std": relative,
            "model": model_name,
        },
    )


def check_predict_schema(
    expected_columns: Sequence[str],
    droppable_columns: Sequence[str],
    incoming: pd.DataFrame,
    expected_dtypes: Optional[Dict[str, str]] = None,
) -> CheckResult:
    """Compare incoming data against the training schema.

    Returns FAIL when a required column is absent. Extra columns are informational
    - they are ignored by the fitted pipeline and cannot corrupt a prediction.
    """
    droppable = set(droppable_columns or ())
    required = [column for column in expected_columns if column not in droppable]
    present = {str(column) for column in incoming.columns}

    missing = [column for column in required if column not in present]
    extra = [column for column in present if column not in set(expected_columns)]

    details: List[str] = []
    dtype_mismatches: Dict[str, str] = {}
    if expected_dtypes:
        for column, trained in expected_dtypes.items():
            if column not in incoming.columns:
                continue
            incoming_kind = _dtype_kind(str(incoming[column].dtype))
            trained_kind = _dtype_kind(str(trained))
            if incoming_kind != trained_kind:
                dtype_mismatches[column] = f"trained={trained_kind}, incoming={incoming_kind}"

    if missing:
        status = FAIL
        summary = f"{len(missing)} required column(s) missing from the incoming data"
        details.append(f"Missing: {', '.join(map(str, missing[:20]))}")
        details.append(
            "Scoring cannot proceed - the model has no values for these features."
        )
    elif dtype_mismatches:
        status = WARN
        summary = f"{len(dtype_mismatches)} column(s) changed type since training"
        for column, message in list(dtype_mismatches.items())[:10]:
            details.append(f"- {column}: {message}")
    else:
        status = PASS
        summary = "incoming schema matches the training schema"

    if extra:
        details.append(f"Extra column(s), ignored: {', '.join(map(str, extra[:10]))}")

    return CheckResult(
        name="predict_schema",
        status=status,
        summary=summary,
        details=details,
        metrics={
            "missing": missing,
            "extra": extra,
            "dtype_mismatches": dtype_mismatches,
            "n_required": len(required),
        },
    )


def _dtype_kind(dtype: str) -> str:
    dtype = str(dtype).lower()
    if any(token in dtype for token in ("int", "float")):
        return "numeric"
    if "bool" in dtype:
        return "boolean"
    if "datetime" in dtype:
        return "datetime"
    return "text"
# ----------------------------------------------------------------------
# Suite runner
# ----------------------------------------------------------------------
def run_validation_suite(
    df: pd.DataFrame,
    target: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    time_series: bool = False,
    leakage_threshold: float = LEAKAGE_THRESHOLD,
) -> ValidationReport:
    """Run every data-level check and return one :class:`ValidationReport`.

    Reproduces the same split ``train`` will use, so drift and duplicate-leakage
    findings describe the run that is about to happen rather than a hypothetical
    one. No model is trained.

    When ``target`` is ``None`` the target-dependent checks are skipped and only
    the structural ones (duplicates, missing data) run.
    """
    report = ValidationReport(
        context={
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "target": target,
            "test_size": test_size,
            "random_state": random_state,
            "time_series": time_series,
        }
    )

    if target is None:
        report.add(check_duplicate_rows(df))
        report.add(
            CheckResult(
                "target_health",
                SKIP,
                "no --target given, so target-dependent checks were skipped",
                details=[
                    "Re-run with --target <column> to check for leakage, imbalance, "
                    "and train/holdout drift."
                ],
            )
        )
        missing_pct = (df.isnull().mean() * 100).to_dict()
        report.add(
            check_missing_data(
                {
                    "missing_pct": missing_pct,
                    "total_missing_pct": float(df.isnull().to_numpy().mean() * 100),
                }
            )
        )
        return report

    from dive.data_intelligence import DataIntelligence

    working = df.dropna(subset=[target]).reset_index(drop=True)
    profile = DataIntelligence(target, random_state).analyze(working)
    report.context["problem_type"] = profile.get("problem_type")
    report.context["n_rows_after_target_dropna"] = int(working.shape[0])

    y = working[target]
    X = working.drop(columns=[target])

    report.add(check_target_health(profile))
    report.add(
        check_target_leakage(
            X, y, profile.get("problem_type", "classification"), threshold=leakage_threshold
        )
    )
    report.add(check_missing_data(profile))

    train_index, test_index = _split_indices(
        working, y, profile.get("problem_type"), test_size, random_state, time_series
    )
    report.add(check_duplicate_rows(X, train_index, test_index))

    if train_index is not None and test_index is not None:
        report.add(
            check_train_holdout_drift(X.loc[train_index], X.loc[test_index])
        )

    return report


def _split_indices(
    frame: pd.DataFrame,
    y: pd.Series,
    problem_type: Optional[str],
    test_size: float,
    random_state: int,
    time_series: bool,
):
    """Reproduce the exact split ``Dive.fit`` will perform."""
    from sklearn.model_selection import train_test_split

    if len(frame) < 10:
        return None, None
    if time_series:
        cutoff = max(1, min(int(len(frame) * (1 - test_size)), len(frame) - 1))
        return frame.index[:cutoff], frame.index[cutoff:]

    stratify = None
    if problem_type == "classification" and y.value_counts().min() >= 2:
        stratify = y
    try:
        train_index, test_index = train_test_split(
            frame.index, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        try:
            train_index, test_index = train_test_split(
                frame.index, test_size=test_size, random_state=random_state
            )
        except ValueError:
            return None, None
    return train_index, test_index


def validate_trained_model(dive: Any, report: Optional[ValidationReport] = None) -> ValidationReport:
    """Add post-training checks (CV stability) to a report.

    Called by ``dive train`` once a winner is known, so the same report object
    carries both the pre-flight data findings and the model-level ones.
    """
    report = report or ValidationReport()
    best_name = getattr(dive, "best_model_name_", None)
    fold_scores = (getattr(dive, "cv_scores_", {}) or {}).get(best_name, [])
    report.add(
        check_cv_stability(
            fold_scores,
            model_name=str(best_name),
            scoring=str(getattr(dive, "scoring", "score")),
        )
    )
    return report
