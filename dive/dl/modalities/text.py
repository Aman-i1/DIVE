"""Text modality for DIVE Deep Learning - ``dive/dl/modalities/text.py``.

Three tiers, chosen by what is installed, and always reported by name so nobody
mistakes tier 3 for tier 1:

1. **Fine-tuned transformer** (``torch`` + ``transformers``) - not a featuriser at
   all but an end-to-end model, so it is reached through
   :func:`finetune_transformer`, which wraps the existing
   :mod:`dive.nlp.transformers` layer rather than reimplementing it.
2. **Frozen sentence embeddings** (``sentence_transformers``) - a pretrained
   encoder used as a fixed featuriser, then the shared MLP head on top.
3. **TF-IDF + truncated SVD** - always available, no optional dependency. This is
   the same representation the NLP domain falls back to, kept identical on
   purpose so ``dive nlp`` and ``dive dl`` numbers are comparable.

Loading reuses :class:`dive.nlp.data.dataset.NLPDataset`, which already resolves
text/target columns, handles headerless CSV/TSV, and reports what it chose.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from dive.dl.config import DLDataConfig, DLTask, Modality
from dive.dl.exceptions import DLBackendError, DLDataError
from dive.dl.modalities.base import ModalityAdapter, RawBatch
from dive.utils.optional import is_available, load_optional

#: Upper bound on SVD components. Enough to carry a document collection's signal
#: into an MLP without making the first Linear layer dominate training time.
_MAX_COMPONENTS = 256

#: Default frozen encoder. Small, CPU-viable, and the same default the NLP
#: embedding layer uses, so the two domains produce comparable vectors.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class TextAdapter(ModalityAdapter):
    """Turns documents into dense vectors for the shared trainer."""

    modality = Modality.TEXT
    accelerated_packages = ("sentence_transformers",)
    fallback_caveat = (
        "Without sentence-transformers, text is represented by TF-IDF + SVD. "
        "That is a solid baseline - it is what 'dive nlp train' uses - but it "
        "carries no pretrained semantics, so expect a few points less accuracy "
        "on nuanced or short-text tasks."
    )
    bytes_per_sample = 8192.0

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(options)
        self._vectorizer: Any = None
        self._reducer: Any = None
        self._encoder: Any = None

    # ------------------------------------------------------------------
    def load(self, source: Any, data_config: DLDataConfig) -> RawBatch:
        from dive.nlp.data.dataset import NLPDataset

        dataset = NLPDataset.from_file(
            source,
            text_column=data_config.input_column,
            target_column=data_config.target_column,
        )
        if len(dataset) == 0:
            raise DLDataError(f"No documents were loaded from '{source}'.")

        notes: List[str] = []
        if dataset.text_column is None:
            notes.append(
                "The text column was resolved positionally; pass --input-col to name it."
            )
        if not dataset.has_labels:
            notes.append(
                "No target column was found, so this corpus is unlabelled. "
                "Name one with --target-col to train a classifier."
            )

        return RawBatch(
            inputs=[str(text) for text in dataset.texts],
            targets=np.asarray(dataset.labels) if dataset.has_labels else None,
            ids=[str(index) for index in range(len(dataset))],
            source=str(source),
            input_column=dataset.text_column,
            target_column=dataset.target_column,
            notes=notes,
        )

    # ------------------------------------------------------------------
    def _texts(self, inputs: Any) -> List[str]:
        if isinstance(inputs, str):
            return [inputs]
        if hasattr(inputs, "tolist"):
            inputs = inputs.tolist()
        if isinstance(inputs, (list, tuple)):
            return [("" if item is None else str(item)) for item in inputs]
        raise DLDataError(
            f"Could not interpret text input of type {type(inputs).__name__}.",
            "Pass a string or a list of strings.",
        )

    def _load_encoder(self) -> Any:
        """Load the frozen sentence encoder, or return ``None`` to use TF-IDF.

        A download failure (no network, gated repo) is a fallback trigger, not an
        error: the run continues on TF-IDF with the reason recorded.
        """
        if not is_available("sentence_transformers"):
            return None
        module = load_optional("sentence_transformers")
        model_name = str(self.option("embedding_model", DEFAULT_EMBEDDING_MODEL))
        try:
            return module.SentenceTransformer(model_name)
        except Exception as exc:
            self.notes.append(
                f"Could not load the '{model_name}' encoder ({type(exc).__name__}); "
                "falling back to TF-IDF + SVD."
            )
            return None

    def fit_transform(self, inputs: Any) -> np.ndarray:
        texts = self._texts(inputs)
        if not texts:
            raise DLDataError("No documents to featurise.")
        self.notes = []

        self._encoder = self._load_encoder()
        if self._encoder is not None:
            matrix = np.asarray(
                self._encoder.encode(texts, show_progress_bar=False), dtype=np.float32
            )
            self.extractor = (
                f"frozen sentence embeddings "
                f"({self.option('embedding_model', DEFAULT_EMBEDDING_MODEL)}, "
                f"{matrix.shape[1]} dims)"
            )
            self.feature_names = [f"emb{index}" for index in range(matrix.shape[1])]
            self.fitted = True
            return matrix

        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        # min_df=1 rather than the NLP default of 2: a DL run is often pointed at
        # a small sample first, and min_df=2 can empty the vocabulary entirely.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=tuple(self.option("ngram_range", (1, 2))),
            max_features=int(self.option("max_features", 30000)),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        sparse = self._vectorizer.fit_transform(texts)
        if sparse.shape[1] == 0:
            raise DLDataError(
                "Every document tokenised to nothing, so there is no vocabulary.",
                "Check that the chosen column really holds text.",
            )

        # SVD needs strictly fewer components than both dimensions.
        components = int(min(_MAX_COMPONENTS, sparse.shape[1] - 1, len(texts) - 1))
        if components < 2:
            dense = np.asarray(sparse.toarray(), dtype=np.float32)
            self._reducer = None
            self.extractor = f"TF-IDF ({dense.shape[1]} terms, corpus too small for SVD)"
            self.feature_names = [f"tfidf{index}" for index in range(dense.shape[1])]
            self.fitted = True
            return dense

        self._reducer = TruncatedSVD(n_components=components, random_state=42)
        matrix = np.asarray(self._reducer.fit_transform(sparse), dtype=np.float32)
        explained = float(getattr(self._reducer, "explained_variance_ratio_", np.array([0.0])).sum())
        self.extractor = (
            f"TF-IDF + SVD ({sparse.shape[1]} terms -> {components} components, "
            f"{explained:.1%} variance)"
        )
        self.feature_names = [f"svd{index}" for index in range(components)]
        self.fitted = True
        return matrix

    def transform(self, inputs: Any) -> np.ndarray:
        if not self.fitted:
            raise DLDataError("The text adapter must be fitted before transform().")
        texts = self._texts(inputs)
        if self._encoder is not None:
            return np.asarray(
                self._encoder.encode(texts, show_progress_bar=False), dtype=np.float32
            )
        sparse = self._vectorizer.transform(texts)
        if self._reducer is None:
            return np.asarray(sparse.toarray(), dtype=np.float32)
        return np.asarray(self._reducer.transform(sparse), dtype=np.float32)

    def coerce(self, data: Any) -> Any:
        return self._texts(data)

    # A loaded SentenceTransformer holds torch tensors and CUDA handles that make
    # a pickled predictor huge and host-specific; the model name is enough to
    # rebuild it on load.
    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_encoder"] = None
        state["_encoder_model_name"] = (
            self.option("embedding_model", DEFAULT_EMBEDDING_MODEL)
            if self.__dict__.get("_encoder") is not None
            else None
        )
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        model_name = state.pop("_encoder_model_name", None)
        self.__dict__.update(state)
        if model_name:
            self.options["embedding_model"] = model_name
            self._encoder = self._load_encoder()
            if self._encoder is None:
                raise DLBackendError(
                    "This model was trained on sentence-transformer embeddings, "
                    f"but the '{model_name}' encoder is unavailable here.",
                    "Install it with 'pip install sentence-transformers', "
                    "or retrain with --no-embeddings.",
                )


def finetune_transformer(
    texts: Sequence[str],
    labels: Sequence[Any],
    task: DLTask = DLTask.CLASSIFICATION,
    model_name: str = "distilbert",
    epochs: int = 3,
    batch_size: int = 16,
    max_seq_length: int = 128,
    learning_rate: float = 2e-5,
    device: Optional[str] = None,
) -> Any:
    """Fine-tune a pretrained transformer end to end.

    Delegates to :mod:`dive.nlp.transformers` - ``TransformerClassifier`` /
    ``TransformerRegressor`` already implement this, and duplicating a training
    loop for the DL domain would give two implementations to keep correct.

    Raises :class:`DLBackendError` when torch or transformers is missing, because
    unlike the featurisers there is no meaningful fallback for *fine-tuning*: the
    caller should use the frozen-embedding or TF-IDF tier instead.
    """
    if not (is_available("torch") and is_available("transformers")):
        missing = [
            name for name in ("torch", "transformers") if not is_available(name)
        ]
        raise DLBackendError(
            "Transformer fine-tuning needs " + " and ".join(missing) + ".",
            "Run 'dive dl doctor' to install them, or drop --arch transformer "
            "to use the frozen-embedding or TF-IDF representation.",
        )

    from dive.nlp.transformers import (
        TransformerClassifier,
        TransformerConfig,
        TransformerRegressor,
    )

    config = TransformerConfig(
        model_name=model_name,
        max_seq_length=int(max_seq_length),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        epochs=int(epochs),
        device=device,
    )
    estimator = (
        TransformerClassifier(config=config)
        if task is DLTask.CLASSIFICATION
        else TransformerRegressor(config=config)
    )
    estimator.fit(list(texts), list(labels))
    return estimator
