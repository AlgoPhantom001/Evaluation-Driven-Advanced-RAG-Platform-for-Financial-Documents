from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.observability.tracing import span

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return not self.indices


class Embedder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._dense: Any | None = None
        self._sparse: Any | None = None
        self._lock = threading.Lock()

    @property
    def dense(self) -> Any:
        if self._dense is None:
            with self._lock:
                if self._dense is None:
                    from fastembed import TextEmbedding

                    with span("load_dense_model", model=self.settings.dense_model):
                        self._dense = TextEmbedding(
                            model_name=self.settings.dense_model,
                            cache_dir=str(self.settings.models_dir),
                        )
                    log.info("dense_model_loaded", model=self.settings.dense_model)
        return self._dense

    @property
    def sparse(self) -> Any:
        if self._sparse is None:
            with self._lock:
                if self._sparse is None:
                    from fastembed import SparseTextEmbedding

                    with span("load_sparse_model", model=self.settings.sparse_model):
                        self._sparse = SparseTextEmbedding(
                            model_name=self.settings.sparse_model,
                            cache_dir=str(self.settings.models_dir),
                        )
                    log.info("sparse_model_loaded", model=self.settings.sparse_model)
        return self._sparse

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.settings.dense_dim), dtype=np.float32)
        with span("embed_documents", count=len(texts)):
            vectors = list(self.dense.embed(list(texts), batch_size=self.settings.embed_batch_size))
        return _normalise(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> NDArray[np.float32]:
        prefixed = f"{self.settings.dense_query_prefix}{text}"
        with span("embed_query"):
            vector = next(iter(self.dense.embed([prefixed])))
        normalised: NDArray[np.float32] = _normalise(
            np.asarray(vector, dtype=np.float32).reshape(1, -1)
        )[0]
        return normalised

    def embed_sparse_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        if not texts:
            return []
        with span("embed_sparse_documents", count=len(texts)):
            raw = list(self.sparse.embed(list(texts), batch_size=self.settings.embed_batch_size))
        return [_to_sparse(item) for item in raw]

    def embed_sparse_query(self, text: str) -> SparseVector:
        with span("embed_sparse_query"):
            raw = next(iter(self.sparse.embed([text])))
        return _to_sparse(raw)

    def warmup(self, *, include_sparse: bool = True) -> None:
        self.embed_query("warmup")
        if include_sparse and self.settings.enable_splade:
            self.embed_sparse_query("warmup")
        log.info("embedder_warm", dense=self.settings.dense_model)


def _normalise(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32)


def _to_sparse(item: Any) -> SparseVector:
    return SparseVector(
        indices=[int(i) for i in item.indices],
        values=[float(v) for v in item.values],
    )
