import os
from typing import Optional

import fsspec

from dreamsboard.engine.storage.dreams_analysis_store.keyval_dreams_analysis_store import KVDreamsAnalysisStore
from dreamsboard.engine.storage.dreams_analysis_store.types import (
    DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME, DEFAULT_PERSIST_PATH
)
from dreamsboard.engine.storage.kvstore.simple_kvstore import SimpleKVStore
from dreamsboard.engine.storage.kvstore.types import BaseInMemoryKVStore
from dreamsboard.engine.utils import concat_dirs


class SimpleDreamsAnalysisStore(KVDreamsAnalysisStore):
    """Simple DreamsAnalysis (Node) store.

    An in-memory store for DreamsAnalysis and Node objects.

    Args:
        simple_kvstore (SimpleKVStore): simple key-value store
        namespace (str): namespace for the template_store

    """

    def __init__(
        self,
        simple_kvstore: Optional[SimpleKVStore] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Init a SimpleDreamsAnalysisStore."""
        simple_kvstore = simple_kvstore or SimpleKVStore()
        super().__init__(simple_kvstore, namespace)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        namespace: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimpleDreamsAnalysisStore":
        """Create a SimpleDreamsAnalysisStore from a persist directory.

        Args:
            persist_dir (str): directory to persist the store
            namespace (Optional[str]): namespace for the template_store
            fs (Optional[fsspec.AbstractFileSystem]): filesystem to use

        """

        if fs is not None:
            persist_path = concat_dirs(persist_dir, DEFAULT_PERSIST_FNAME)
        else:
            persist_path = os.path.join(persist_dir, DEFAULT_PERSIST_FNAME)
        return cls.from_persist_path(persist_path, namespace=namespace, fs=fs)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        namespace: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimpleDreamsAnalysisStore":
        """Create a SimpleDreamsAnalysisStore from a persist path.

        Args:
            persist_path (str): Path to persist the store
            namespace (Optional[str]): namespace for the template_store
            fs (Optional[fsspec.AbstractFileSystem]): filesystem to use

        """

        simple_kvstore = SimpleKVStore.from_persist_path(persist_path, fs=fs)
        return cls(simple_kvstore, namespace)

    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the store."""
        if isinstance(self._kvstore, BaseInMemoryKVStore):
            self._kvstore.persist(persist_path, fs=fs)

    @classmethod
    def from_dict(
        cls, save_dict: dict, namespace: Optional[str] = None
    ) -> "SimpleDreamsAnalysisStore":
        simple_kvstore = SimpleKVStore.from_dict(save_dict)
        return cls(simple_kvstore, namespace)

    def to_dict(self) -> dict:
        assert isinstance(self._kvstore, SimpleKVStore)
        return self._kvstore.to_dict()


# alias for backwards compatibility
DreamsAnalysisStore = SimpleDreamsAnalysisStore
