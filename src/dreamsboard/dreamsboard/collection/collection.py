from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass
class QueryResult:
    content: str
    score: float
    metadata: Dict[str, Any]


class BaseCollection:
    """Unified collection interface that hides different data sources."""

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        raise NotImplementedError

    def query(self, query: str, top_k: int = 5) -> List[QueryResult]:
        raise NotImplementedError


_COLLECTION_REGISTRY: Dict[str, Callable[..., BaseCollection]] = {}


def register_collection(name: str, factory: Callable[..., BaseCollection]) -> None:
    _COLLECTION_REGISTRY[name] = factory


def create_collection(mode: str, **kwargs: Any) -> BaseCollection:
    try:
        factory = _COLLECTION_REGISTRY[mode]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported data_base: {mode}") from exc
    return factory(**kwargs)
