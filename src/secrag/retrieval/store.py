from __future__ import annotations

import shutil
import threading
import uuid
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from secrag.core.config import Settings, get_settings
from secrag.core.errors import RetrievalError
from secrag.core.logging import get_logger
from secrag.core.types import Chunk, FilingSection, ScoredChunk
from secrag.observability.tracing import span
from secrag.retrieval.embedder import Embedder, SparseVector

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

log = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "splade"
COLLECTION = "filings"

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class SearchFilter:
    def __init__(
        self,
        *,
        tickers: Sequence[str] = (),
        fiscal_years: Sequence[int] = (),
        sections: Sequence[FilingSection] = (),
    ) -> None:
        self.tickers = [t.upper() for t in tickers if t]
        self.fiscal_years = list(fiscal_years)
        self.sections = list(sections)

    @property
    def is_empty(self) -> bool:
        return not (self.tickers or self.fiscal_years or self.sections)

    def to_qdrant(self) -> Any | None:
        if self.is_empty:
            return None
        from qdrant_client import models

        conditions: list[Any] = []
        if self.tickers:
            conditions.append(
                models.FieldCondition(key="ticker", match=models.MatchAny(any=self.tickers))
            )
        if self.fiscal_years:
            conditions.append(
                models.FieldCondition(
                    key="fiscal_year", match=models.MatchAny(any=list(self.fiscal_years))
                )
            )
        if self.sections:
            conditions.append(
                models.FieldCondition(
                    key="section", match=models.MatchAny(any=[s.value for s in self.sections])
                )
            )
        return models.Filter(must=conditions)

    def matches(self, chunk: Chunk) -> bool:
        if self.tickers and (chunk.ticker or "").upper() not in self.tickers:
            return False
        if self.fiscal_years and chunk.fiscal_year not in self.fiscal_years:
            return False
        return not (self.sections and chunk.section not in self.sections)


class VectorStore:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings)
        self.path = self.settings.index_dir / "qdrant"
        self._client: Any | None = None
        self._lock = threading.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from qdrant_client import QdrantClient

                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._client = QdrantClient(path=str(self.path))
                    self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        assert self._client is not None
        if self._client.collection_exists(COLLECTION):
            return
        self._client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=self.settings.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
            for field, schema in (
                ("ticker", models.PayloadSchemaType.KEYWORD),
                ("section", models.PayloadSchemaType.KEYWORD),
                ("fiscal_year", models.PayloadSchemaType.INTEGER),
                ("filing_id", models.PayloadSchemaType.KEYWORD),
            ):
                try:
                    self._client.create_payload_index(
                        COLLECTION, field_name=field, field_schema=schema
                    )
                except Exception as exc:
                    log.debug("payload_index_skipped", field=field, error=str(exc))
        log.info("collection_created", collection=COLLECTION, dim=self.settings.dense_dim)

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def reset(self) -> None:
        self.close()
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        log.info("index_reset", path=str(self.path))

    def upsert(self, chunks: Sequence[Chunk], *, with_sparse: bool = True) -> int:
        if not chunks:
            return 0
        from qdrant_client import models

        texts = [c.contextual_text() for c in chunks]
        with span("upsert", count=len(chunks)):
            dense = self.embedder.embed_documents(texts)
            sparse: list[SparseVector] | None = None
            if with_sparse and self.settings.enable_splade:
                sparse = self.embedder.embed_sparse_documents(texts)

            points: list[Any] = []
            for i, chunk in enumerate(chunks):
                vectors: dict[str, Any] = {DENSE_VECTOR: dense[i].tolist()}
                if sparse is not None and not sparse[i].is_empty:
                    vectors[SPARSE_VECTOR] = models.SparseVector(
                        indices=sparse[i].indices, values=sparse[i].values
                    )
                points.append(
                    models.PointStruct(
                        id=_point_id(chunk.chunk_id),
                        vector=vectors,
                        payload=chunk.model_dump(mode="json"),
                    )
                )

            for batch_start in range(0, len(points), 256):
                self.client.upsert(COLLECTION, points=points[batch_start : batch_start + 256])

        log.info("upserted", chunks=len(chunks), sparse=sparse is not None)
        return len(chunks)

    def count(self) -> int:
        try:
            return int(self.client.count(COLLECTION, exact=True).count)
        except Exception:
            return 0

    def iter_chunks(self, batch: int = 512) -> Iterable[Chunk]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                COLLECTION, limit=batch, offset=offset, with_payload=True, with_vectors=False
            )
            for point in points:
                if point.payload:
                    yield Chunk.model_validate(point.payload)
            if offset is None:
                break

    def search_dense(
        self, vector: NDArray[np.float32], limit: int, flt: SearchFilter | None = None
    ) -> list[ScoredChunk]:
        with span("search_dense", limit=limit):
            response = self.client.query_points(
                COLLECTION,
                query=vector.tolist(),
                using=DENSE_VECTOR,
                limit=limit,
                query_filter=flt.to_qdrant() if flt else None,
                with_payload=True,
            )
        return _to_scored(response.points, stage="dense")

    def search_sparse(
        self, vector: SparseVector, limit: int, flt: SearchFilter | None = None
    ) -> list[ScoredChunk]:
        if vector.is_empty:
            return []
        from qdrant_client import models

        with span("search_sparse", limit=limit):
            response = self.client.query_points(
                COLLECTION,
                query=models.SparseVector(indices=vector.indices, values=vector.values),
                using=SPARSE_VECTOR,
                limit=limit,
                query_filter=flt.to_qdrant() if flt else None,
                with_payload=True,
            )
        return _to_scored(response.points, stage="splade")


def _to_scored(points: Sequence[Any], *, stage: str) -> list[ScoredChunk]:
    out: list[ScoredChunk] = []
    for rank, point in enumerate(points, start=1):
        if not point.payload:
            continue
        try:
            chunk = Chunk.model_validate(point.payload)
        except Exception as exc:
            msg = "Stored payload could not be parsed back into a Chunk"
            raise RetrievalError(msg, detail=str(exc)) from exc
        score = float(point.score)
        out.append(
            ScoredChunk(
                chunk=chunk, score=score, stage=stage, rank=rank, component_scores={stage: score}
            )
        )
    return out
