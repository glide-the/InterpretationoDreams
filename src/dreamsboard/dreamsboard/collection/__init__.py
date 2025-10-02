from .collection import BaseCollection, QueryResult, create_collection, register_collection
from .local_collection import LocalCollection  # noqa: F401
from .paper_collection import PaperCollection  # noqa: F401
from .web_collection import WebCollection  # noqa: F401

__all__ = [
    "BaseCollection",
    "QueryResult",
    "create_collection",
    "register_collection",
    "LocalCollection",
    "PaperCollection",
    "WebCollection",
]
