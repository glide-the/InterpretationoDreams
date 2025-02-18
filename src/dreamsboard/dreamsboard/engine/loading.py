import logging
from typing import Any, List, Optional, Sequence

from dreamsboard.engine.engine_builder import BaseEngineBuilder
from dreamsboard.engine.registry import INDEX_STRUCT_TYPE_TO_INDEX_CLASS
from dreamsboard.engine.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)


def load_store_from_storage(
    storage_context: StorageContext,
    index_id: Optional[str] = None,
    **kwargs: Any,
) -> BaseEngineBuilder:
    """Load index from storage context.

    Args:
        :param index_id: index_id(Optional[Sequence[str]]): IDs of the indices to load.
            Defaults to None, which loads all indices in the index store.
        :param storage_context: storage_context (StorageContext): storage context containing
            template store, index store

        **kwargs: Additional keyword args to pass to the index constructors.
    """
    index_ids: Optional[Sequence[str]]
    if index_id is None:
        index_ids = None
    else:
        index_ids = [index_id]

    indices = load_indices_from_storage(storage_context, index_ids=index_ids, **kwargs)

    if len(indices) == 0:
        raise ValueError(
            "No index in storage context, check if you specified the right persist_dir."
        )
    elif len(indices) > 1:
        raise ValueError(
            f"Expected to load a single index, but got {len(indices)} instead. "
            "Please specify index_id."
        )

    return indices[0]


def load_indices_from_storage(
    storage_context: StorageContext,
    index_ids: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> List[BaseEngineBuilder]:
    """Load multiple indices from storage context

    Args:
        :param storage_context: storage_context (StorageContext): storage context containing
            template store, index store
        :param index_ids: index_id(Optional[Sequence[str]]): IDs of the indices to load.
            Defaults to None, which loads all indices in the index store.

        **kwargs: Additional keyword args to pass to the index constructors.
    """
    if index_ids is None:
        logger.info("Loading all indices.")
        index_structs = storage_context.index_store.index_structs()
    else:
        logger.info(f"Loading indices with ids: {index_ids}")
        index_structs = []
        for index_id in index_ids:
            index_struct = storage_context.index_store.get_index_struct(index_id)
            if index_struct is None:
                raise ValueError(f"Failed to load index with ID {index_id}")
            index_structs.append(index_struct)

    indices = []
    for index_struct in index_structs:
        type_ = index_struct.get_type()
        index_cls = INDEX_STRUCT_TYPE_TO_INDEX_CLASS[type_]
        index = index_cls(
            index_struct=index_struct, storage_context=storage_context, **kwargs
        )
        indices.append(index)
    return indices
