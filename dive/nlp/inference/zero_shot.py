"""Zero-Shot Text Classification Engine - `dive/nlp/inference/zero_shot.py`.

Provides zero-shot text classification without requiring labeled training datasets.
Computes semantic similarity vectors between input text documents and candidate label
hypotheses in continuous semantic projection space with calibrated probability distributions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dive.nlp.features.lsa import LSARepresentation
from dive.utils.optional import is_available, load_optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

DEFAULT_ANCHORS: Dict[str, str] = {
    "finance": "finance money banking payment billing invoice account investment dividend tax cash credit loan revenue profit debt budget interest transaction stock",
    "banking": "banking bank account deposit withdrawal loan transfer credit debit savings wire interest",
    "sports": "sports game match player team tournament score coach win league champion football basketball soccer baseball athlete championship",
    "technology": "technology software code algorithm hardware computer programming engineering digital server data network cloud system application developer",
    "science": "science biology chemistry physics experiment research laboratory study theory scientist scientific discovery astronomy genetics",
    "medical": "medical health disease patient doctor hospital medicine clinical symptoms treatment surgery diagnosis prescription healthcare drug therapy",
    "health": "health wellness fitness medical diet nutrition symptom hospital doctor physician exercise clinic healthcare",
    "politics": "politics government election president policy voting law parliament senator campaign diplomacy congress leader democracy legislation minister",
    "entertainment": "entertainment movie film music concert actor song drama celebrity cinema theatre show performance festival television hollywood",
    "spam": "spam free offer win prize urgent click here claim cash discount lottery reward credit promo buy now guaranteed gift call now",
    "business": "business corporate enterprise commerce trade strategy market sales management company partnership client deal industry revenue executive",
    "education": "education school university college student teacher study degree learning course classroom exam tuition academy professor lecture",
    "support": "support service issue help ticket problem account password bug refund error assist troubleshooting question fix cancel inquiry",
    "travel": "travel trip vacation flight hotel tourism airline booking destination passport airport luggage journey visit hotel resort",
    "real estate": "real estate property house apartment rent lease mortgage home realtor tenant landlord buying selling listing residential",
    "food": "food cooking recipe restaurant meal dining chef dish culinary kitchen dinner lunch breakfast beverage gourmet cuisine",
}


class ZeroShotClassifier:
    """Zero-shot classification via continuous semantic projection and similarity calibration."""

    def __init__(
        self,
        temperature: float = 0.20,
        hypothesis_template: str = "This document is about {}.",
        custom_anchors: Optional[Dict[str, str]] = None,
    ) -> None:
        self.temperature = max(0.01, float(temperature))
        self.hypothesis_template = hypothesis_template
        self.custom_anchors = custom_anchors or {}
        self._st_model: Any = None

        if is_available("sentence_transformers"):
            try:
                st = load_optional("sentence_transformers")
                self._st_model = st.SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._st_model = None

    def _expand_hypothesis(self, label: str) -> str:
        lbl_clean = label.strip()
        lbl_lower = lbl_clean.lower()
        expanded_terms: List[str] = []

        anchors_map = {**DEFAULT_ANCHORS, **self.custom_anchors}
        for key, terms in anchors_map.items():
            if key in lbl_lower or lbl_lower in key:
                expanded_terms.append(terms)

        base_hyp = self.hypothesis_template.format(lbl_clean)
        if expanded_terms:
            unique_terms = " ".join(dict.fromkeys(" ".join(expanded_terms).split()))
            return f"{base_hyp} Related keywords: {unique_terms}."
        return base_hyp

    def predict(
        self,
        texts: Union[str, Sequence[str]],
        candidate_labels: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Classify inputs into candidate_labels with calibrated confidence scores."""
        if isinstance(texts, str):
            docs = [texts]
        else:
            docs = list(texts)

        if not candidate_labels:
            raise ValueError("At least one candidate label must be provided.")

        labels = [str(lbl).strip() for lbl in candidate_labels if str(lbl).strip()]
        if not labels:
            raise ValueError("Candidate labels cannot be empty.")

        # Expand labels into semantic hypothesis sentences with domain anchors
        hypotheses = [self._expand_hypothesis(lbl) for lbl in labels]

        if self._st_model is not None:
            doc_vectors = self._st_model.encode(docs, normalize_embeddings=True)
            label_vectors = self._st_model.encode(hypotheses, normalize_embeddings=True)
            sim_matrix = cosine_similarity(doc_vectors, label_vectors)
        else:
            # Dual-granularity representation: stop-word filtered word n-grams + subword character n-grams
            word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), sublinear_tf=True)
            corpus = list(docs) + list(hypotheses) + labels
            word_vec.fit(corpus)
            char_vec.fit(corpus)

            w_sim = cosine_similarity(word_vec.transform(docs), word_vec.transform(hypotheses))
            c_sim = cosine_similarity(char_vec.transform(docs), char_vec.transform(hypotheses))
            # 70% word semantics + 30% subword morphological matching
            sim_matrix = 0.70 * w_sim + 0.30 * c_sim

        results = []
        for i, text in enumerate(docs):
            row = np.asarray(sim_matrix[i], dtype=float)
            max_val = float(np.max(row))
            min_val = float(np.min(row))
            spread = max_val - min_val

            if spread > 1e-6:
                # Dynamically adapt effective temperature to representation dispersion
                eff_temp = max(0.01, min(self.temperature, spread * 0.40))
                scaled_row = (row - max_val) / eff_temp
                exp_row = np.exp(scaled_row)
                probs = exp_row / np.sum(exp_row)
            else:
                probs = np.full_like(row, 1.0 / len(labels))

            doc_probs = {labels[j]: float(probs[j]) for j in range(len(labels))}
            sorted_pairs = sorted(doc_probs.items(), key=lambda x: x[1], reverse=True)
            top_label, top_score = sorted_pairs[0]
            results.append({
                "text": text,
                "predicted_label": top_label,
                "confidence": top_score,
                "probabilities": doc_probs,
                "ranking": sorted_pairs,
            })

        return results


