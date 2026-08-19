"""NLP Production Drift Monitoring & Telemetry - `dive/nlp/monitoring/drift.py`.

Monitors production NLP text streams against baseline reference data:
- Token and character length distribution shift (Kolmogorov-Smirnov & Wasserstein metrics)
- Vocabulary expansion and Out-Of-Vocabulary (OOV) rate tracking
- Prediction class distribution shift (Population Stability Index - PSI)
- Actionable diagnostic alerting and ASCII reporting
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy import stats


_WORD_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


@dataclass
class NLPDriftReport:
    """Diagnostic report capturing text drift and prediction distribution shift."""

    reference_samples: int
    current_samples: int
    length_drift: Dict[str, Any]
    vocabulary_drift: Dict[str, Any]
    prediction_drift: Optional[Dict[str, Any]] = None
    drift_detected: bool = False
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert drift report to serializable dictionary."""
        return {
            "reference_samples": self.reference_samples,
            "current_samples": self.current_samples,
            "drift_detected": self.drift_detected,
            "alerts": self.alerts,
            "length_drift": self.length_drift,
            "vocabulary_drift": self.vocabulary_drift,
            "prediction_drift": self.prediction_drift,
        }

    def render(self) -> str:
        """Render formatted ASCII diagnostic report."""
        status_str = "DRIFT DETECTED" if self.drift_detected else "STABLE (NO DRIFT)"
        lines = [
            "=" * 85,
            "                      DIVE NLP DRIFT & MONITORING REPORT                      ",
            "=" * 85,
            f"Status                   : {status_str}",
            f"Reference Sample Size    : {self.reference_samples}",
            f"Current Sample Size      : {self.current_samples}",
            "-" * 85,
            "1. DOCUMENT LENGTH DRIFT",
        ]

        # Length drift metrics
        tok_stat = self.length_drift.get("token_length", {})
        lines.append(
            f"  - Token Length KS p-val: {tok_stat.get('p_value', 1.0):.4f} "
            f"(Wasserstein Dist: {tok_stat.get('wasserstein_dist', 0.0):.2f}, Shift: {tok_stat.get('drift_detected', False)})"
        )
        lines.append(
            f"  - Ref Avg Tokens       : {tok_stat.get('ref_mean', 0.0):.1f} ± {tok_stat.get('ref_std', 0.0):.1f} | "
            f"Curr Avg Tokens: {tok_stat.get('curr_mean', 0.0):.1f} ± {tok_stat.get('curr_std', 0.0):.1f}"
        )

        # Vocabulary drift
        lines.append("-" * 85)
        lines.append("2. VOCABULARY & OOV SHIFT")
        voc = self.vocabulary_drift
        lines.append(f"  - Ref Vocab Size       : {voc.get('ref_vocab_size', 0)}")
        lines.append(f"  - Curr Vocab Size      : {voc.get('curr_vocab_size', 0)}")
        lines.append(
            f"  - Out-of-Vocab (OOV)   : {voc.get('oov_tokens_count', 0)} tokens "
            f"(Rate: {voc.get('oov_rate', 0.0):.2%}, Drift: {voc.get('drift_detected', False)})"
        )
        top_emergent = voc.get("top_emergent_words", [])
        if top_emergent:
            lines.append(f"  - Top Emergent Words   : {', '.join(top_emergent[:8])}")

        # Prediction drift
        if self.prediction_drift:
            lines.append("-" * 85)
            lines.append("3. PREDICTION DISTRIBUTION SHIFT")
            pd_stat = self.prediction_drift
            lines.append(
                f"  - Population Stability : PSI = {pd_stat.get('psi', 0.0):.4f} "
                f"({pd_stat.get('shift_level', 'None')}, Drift: {pd_stat.get('drift_detected', False)})"
            )

        # Alerts
        if self.alerts:
            lines.append("-" * 85)
            lines.append("4. ACTIONABLE ALERTS")
            for alert in self.alerts:
                lines.append(f"  [!] {alert}")

        lines.append("=" * 85)
        return "\n".join(lines)


class NLPDriftMonitor:
    """Production monitor tracking text distribution and prediction drift."""

    def __init__(
        self,
        reference_texts: Sequence[str],
        reference_predictions: Optional[Sequence[Any]] = None,
        p_value_threshold: float = 0.05,
        oov_threshold: float = 0.15,
        psi_threshold: float = 0.25,
    ) -> None:
        self.p_value_threshold = p_value_threshold
        self.oov_threshold = oov_threshold
        self.psi_threshold = psi_threshold

        self.ref_texts = [str(t) for t in reference_texts]
        self.ref_predictions = list(reference_predictions) if reference_predictions is not None else None

        # Extract reference features
        self.ref_token_lengths = np.array([len(_WORD_TOKEN_PATTERN.findall(t)) for t in self.ref_texts])
        self.ref_char_lengths = np.array([len(t) for t in self.ref_texts])

        # Reference vocabulary
        self.ref_vocabulary: Set[str] = set()
        for t in self.ref_texts:
            self.ref_vocabulary.update(w.lower() for w in _WORD_TOKEN_PATTERN.findall(t))

    def check_drift(
        self,
        current_texts: Sequence[str],
        current_predictions: Optional[Sequence[Any]] = None,
    ) -> NLPDriftReport:
        """Evaluate current production text batch against reference distribution."""
        curr_texts = [str(t) for t in current_texts]
        if not curr_texts:
            raise ValueError("Current text stream is empty. Cannot compute drift.")

        alerts: List[str] = []
        drift_detected = False

        # -------------------------------------------------------------
        # 1. Document Length Drift (Tokens & Chars)
        # -------------------------------------------------------------
        curr_token_lengths = np.array([len(_WORD_TOKEN_PATTERN.findall(t)) for t in curr_texts])
        curr_char_lengths = np.array([len(t) for t in curr_texts])

        # KS test & Wasserstein distance for token lengths
        ks_tok = stats.ks_2samp(self.ref_token_lengths, curr_token_lengths)
        wass_tok = float(stats.wasserstein_distance(self.ref_token_lengths, curr_token_lengths))
        tok_drift = bool(ks_tok.pvalue < self.p_value_threshold and wass_tok > 2.0)

        if tok_drift:
            drift_detected = True
            alerts.append(
                f"Token length drift detected (p={ks_tok.pvalue:.4f}, distance={wass_tok:.2f}). "
                f"Ref avg: {np.mean(self.ref_token_lengths):.1f}, Curr avg: {np.mean(curr_token_lengths):.1f}."
            )

        # KS test for character lengths
        ks_char = stats.ks_2samp(self.ref_char_lengths, curr_char_lengths)
        wass_char = float(stats.wasserstein_distance(self.ref_char_lengths, curr_char_lengths))
        char_drift = bool(ks_char.pvalue < self.p_value_threshold and wass_char > 15.0)

        length_drift = {
            "token_length": {
                "p_value": float(ks_tok.pvalue),
                "wasserstein_dist": wass_tok,
                "ref_mean": float(np.mean(self.ref_token_lengths)),
                "ref_std": float(np.std(self.ref_token_lengths)),
                "curr_mean": float(np.mean(curr_token_lengths)),
                "curr_std": float(np.std(curr_token_lengths)),
                "drift_detected": tok_drift,
            },
            "char_length": {
                "p_value": float(ks_char.pvalue),
                "wasserstein_dist": wass_char,
                "ref_mean": float(np.mean(self.ref_char_lengths)),
                "ref_std": float(np.std(self.ref_char_lengths)),
                "curr_mean": float(np.mean(curr_char_lengths)),
                "curr_std": float(np.std(curr_char_lengths)),
                "drift_detected": char_drift,
            },
        }

        # -------------------------------------------------------------
        # 2. Vocabulary & Out-of-Vocabulary (OOV) Rate Shift
        # -------------------------------------------------------------
        curr_tokens: List[str] = []
        for t in curr_texts:
            curr_tokens.extend(w.lower() for w in _WORD_TOKEN_PATTERN.findall(t))

        total_curr_tokens = len(curr_tokens)
        curr_vocab = set(curr_tokens)

        oov_tokens = [tok for tok in curr_tokens if tok not in self.ref_vocabulary]
        oov_count = len(oov_tokens)
        oov_rate = float(oov_count / max(1, total_curr_tokens))

        oov_drift = bool(oov_rate > self.oov_threshold)
        if oov_drift:
            drift_detected = True
            alerts.append(
                f"Elevated Out-of-Vocabulary (OOV) rate detected: {oov_rate:.2%} (threshold: {self.oov_threshold:.2%}). "
                "Significant domain vocabulary shift observed."
            )

        # Count top emergent words
        emergent_counts: Dict[str, int] = {}
        for w in oov_tokens:
            emergent_counts[w] = emergent_counts.get(w, 0) + 1
        top_emergent = [
            k for k, _ in sorted(emergent_counts.items(), key=lambda item: item[1], reverse=True)[:10]
        ]

        vocabulary_drift = {
            "ref_vocab_size": len(self.ref_vocabulary),
            "curr_vocab_size": len(curr_vocab),
            "total_current_tokens": total_curr_tokens,
            "oov_tokens_count": oov_count,
            "oov_rate": round(oov_rate, 4),
            "top_emergent_words": top_emergent,
            "drift_detected": oov_drift,
        }

        # -------------------------------------------------------------
        # 3. Prediction Distribution Shift (PSI)
        # -------------------------------------------------------------
        prediction_drift = None
        if self.ref_predictions is not None and current_predictions is not None:
            curr_preds = list(current_predictions)
            ref_preds = self.ref_predictions

            # Compute categorical PSI
            all_classes = sorted(list(set(ref_preds).union(set(curr_preds))))
            eps = 1e-4

            ref_counts = {c: ref_preds.count(c) for c in all_classes}
            curr_counts = {c: curr_preds.count(c) for c in all_classes}

            ref_total = len(ref_preds)
            curr_total = len(curr_preds)

            psi = 0.0
            for c in all_classes:
                p_ref = (ref_counts[c] / ref_total) + eps
                p_curr = (curr_counts[c] / curr_total) + eps
                psi += (p_curr - p_ref) * np.log(p_curr / p_ref)

            psi = float(psi)
            pred_drift = bool(psi >= self.psi_threshold)

            if pred_drift:
                drift_detected = True
                alerts.append(
                    f"Prediction class distribution drift detected: PSI = {psi:.4f} "
                    f"(threshold: {self.psi_threshold:.2f})."
                )

            shift_level = "Significant" if psi >= 0.25 else ("Moderate" if psi >= 0.10 else "Negligible")

            prediction_drift = {
                "psi": round(psi, 4),
                "shift_level": shift_level,
                "ref_distribution": {str(k): round(v / ref_total, 4) for k, v in ref_counts.items()},
                "curr_distribution": {str(k): round(v / curr_total, 4) for k, v in curr_counts.items()},
                "drift_detected": pred_drift,
            }

        return NLPDriftReport(
            reference_samples=len(self.ref_texts),
            current_samples=len(curr_texts),
            length_drift=length_drift,
            vocabulary_drift=vocabulary_drift,
            prediction_drift=prediction_drift,
            drift_detected=drift_detected,
            alerts=alerts,
        )


def monitor_nlp_drift(
    reference_texts: Sequence[str],
    current_texts: Sequence[str],
    reference_predictions: Optional[Sequence[Any]] = None,
    current_predictions: Optional[Sequence[Any]] = None,
    oov_threshold: float = 0.15,
) -> NLPDriftReport:
    """Convenience helper to compute an NLPDriftReport between two document batches."""
    monitor = NLPDriftMonitor(
        reference_texts=reference_texts,
        reference_predictions=reference_predictions,
        oov_threshold=oov_threshold,
    )
    return monitor.check_drift(
        current_texts=current_texts,
        current_predictions=current_predictions,
    )
