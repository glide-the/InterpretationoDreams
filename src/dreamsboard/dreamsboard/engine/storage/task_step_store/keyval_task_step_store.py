"""analysis store."""

from typing import Dict, Optional, Sequence

from dreamsboard.engine.entity.task_step.task_step import TaskStepNode
from dreamsboard.engine.generate.code_generate import (
    CodeGenerator
)
from dreamsboard.engine.storage.task_step_store.types import BaseTaskStepStore, RefTaskStepInfo
from dreamsboard.engine.storage.task_step_store.utils import json_to_task_step, task_step_to_json

from dreamsboard.engine.storage.kvstore.types import BaseKVStore

DEFAULT_NAMESPACE = "task_step_store"


class KVTaskStepStore(BaseTaskStepStore):
    """task_step (Node) store.

    NOTE: at the moment, this store is primarily used to store TaskStepNode objects.
    Each node will be assigned an ID.

    The same task_stepstore can be reused across index structures. This
    allows you to reuse the same storage for multiple index structures;
    otherwise, each index would create a task_stepstore under the hood.

    .. code-block:: python
            store = SimpleTaskStepStore()
            task_step = TaskStepNode.from_config(cfg={
                "task_step_guidance_context": "task_step_guidance",
                "task_step_personality_context": "task_step_personality"
            })
            store.add_task_step([task_step])
            logger.info(store.task_step_all)
            task_step_store_path = concat_dirs(dirname="./storage", basename="task_step_store.json")
            store.persist(persist_path=task_step_store_path)

    This will use the same task_stepstore for multiple index structures.

    Args:
        kvstore (BaseKVStore): key-value store
        namespace (str): namespace for the task_stepstore

    """

    def __init__(
            self,
            kvstore: BaseKVStore,
            namespace: Optional[str] = None,
    ) -> None:
        """Init a KVtask_stepStore."""
        self._kvstore = kvstore
        self._namespace = namespace or DEFAULT_NAMESPACE
        self._node_collection = f"{self._namespace}/data"
        self._ref_task_step_collection = f"{self._namespace}/ref_task_step_info"
        self._metadata_collection = f"{self._namespace}/metadata"

    @property
    def task_step_all(self) -> Dict[str, TaskStepNode]:
        """Get all task_step.

        Returns:
            Dict[str, TaskStepNode]: task_step

        """
        json_dict = self._kvstore.get_all(collection=self._node_collection)
        return {key: json_to_task_step(json) for key, json in json_dict.items()}

    def add_task_step(
            self, task_steps: Sequence[TaskStepNode], allow_update: bool = True
    ) -> None:
        """Add a task_steps to the store.

        Args:
            task_steps (List[TaskStepNode]): task_steps
            allow_update (bool): allow update of store from task_steps
            :param allow_update:
            :param task_steps:

        """
        for node in task_steps:
            # NOTE: task_step could already exist in the store, but we overwrite it
            if not allow_update and self.task_step_exists(node.node_id):
                raise ValueError(
                    f"node_id {node.node_id} already exists. "
                    "Set allow_update to True to overwrite."
                )
            node_key = node.node_id
            data = task_step_to_json(node)
            self._kvstore.put(node_key, data, collection=self._node_collection)

            # update analysis_collection if needed
            metadata = {"task_step_hash": node.hash}
            if isinstance(node, TaskStepNode) and node.ref_task_step_id is not None:
                ref_task_step_info = self.get_ref_task_step_info(node.ref_task_step_id) or RefTaskStepInfo()
                ref_task_step_info.node_ids.append(node.node_id)
                if not ref_task_step_info.metadata:
                    ref_task_step_info.metadata = node.metadata or {}
                self._kvstore.put(
                    node.ref_task_step_id,
                    ref_task_step_info.to_dict(),
                    collection=self._ref_task_step_collection,
                )

                # update metadata with map
                metadata["ref_task_step_id"] = node.ref_task_step_id
                self._kvstore.put(
                    node_key, metadata, collection=self._metadata_collection
                )
            else:
                self._kvstore.put(
                    node_key, metadata, collection=self._metadata_collection
                )

    def get_task_step(self, task_step_id: str, raise_error: bool = True) -> Optional[TaskStepNode]:
        """Get a task_step from the store.

        Args:
            task_step_id (str): task_step id
            raise_error (bool): raise error if task_step_id not found

        """
        json = self._kvstore.get(task_step_id, collection=self._node_collection)
        if json is None:
            if raise_error:
                raise ValueError(f"task_step_id {task_step_id} not found.")
            else:
                return None
        return json_to_task_step(json)

    def delete_task_step(
            self, task_step_id: str, raise_error: bool = True, remove_ref_task_step_node: bool = True
    ) -> None:
        """Delete a task_step from the store."""
        if remove_ref_task_step_node:
            self._remove_ref_task_step_node(task_step_id)

        delete_success = self._kvstore.delete(task_step_id, collection=self._node_collection)
        _ = self._kvstore.delete(task_step_id, collection=self._metadata_collection)

        if not delete_success and raise_error:
            raise ValueError(f"task_step_id {task_step_id} not found.")

    def task_step_exists(self, task_step_id: str) -> bool:
        """Check if task_step exists."""
        return self._kvstore.get(task_step_id, collection=self._node_collection) is not None

    def get_ref_task_step_info(self, ref_task_step_id: str) -> Optional[RefTaskStepInfo]:
        """Get the RefTaskStepInfo for a given ref_task_step_id."""
        ref_task_step_info = self._kvstore.get(
            ref_task_step_id, collection=self._ref_task_step_collection
        )
        if not ref_task_step_info:
            return None

        # TODO: deprecated legacy support
        if "task_step_ids" in ref_task_step_info:
            ref_task_step_info["node_ids"] = ref_task_step_info.get("task_step_ids", [])
            ref_task_step_info.pop("task_step_ids")

            ref_task_step_info["metadata"] = ref_task_step_info.get("extra_info", {})
            ref_task_step_info.pop("extra_info")

        return RefTaskStepInfo(**ref_task_step_info)

    def get_all_ref_task_step_info(self) -> Optional[Dict[str, RefTaskStepInfo]]:
        """Get a mapping of ref_task_step_id -> RefTaskStepInfo for all ingested nodes."""
        ref_task_step_infos = self._kvstore.get_all(collection=self._ref_task_step_collection)
        if ref_task_step_infos is None:
            return None

        # TODO: deprecated legacy support
        all_ref_task_step_infos = {}
        for task_step_id, ref_task_step_info in ref_task_step_infos.items():
            if "task_step_ids" in ref_task_step_info:
                ref_task_step_info["node_ids"] = ref_task_step_info.get("task_step_ids", [])
                ref_task_step_info.pop("task_step_ids")

                ref_task_step_info["metadata"] = ref_task_step_info.get("extra_info", {})
                ref_task_step_info.pop("extra_info")
                all_ref_task_step_infos[task_step_id] = RefTaskStepInfo(**ref_task_step_info)

        return all_ref_task_step_infos

    def ref_task_step_exists(self, ref_task_step_id: str) -> bool:
        """Check if a ref_task_step_id has been ingested."""
        return self.get_ref_task_step_info(ref_task_step_id) is not None

    def _remove_ref_task_step_node(self, task_step_id: str) -> None:
        """Helper function to remove node task_step_id from ref_task_step_collection."""
        metadata = self._kvstore.get(task_step_id, collection=self._metadata_collection)
        if metadata is None:
            return

        ref_task_step_id = metadata.get("ref_task_step_id", None)

        if ref_task_step_id is None:
            return

        ref_task_step_info = self._kvstore.get(
            ref_task_step_id, collection=self._ref_task_step_collection
        )

        if ref_task_step_info is not None:
            ref_task_step_obj = RefTaskStepInfo(**ref_task_step_info)

            ref_task_step_obj.node_ids.remove(task_step_id)

            # delete ref_task_step from collection if it has no more task_step_ids
            if len(ref_task_step_obj.node_ids) > 0:
                self._kvstore.put(
                    ref_task_step_id,
                    ref_task_step_obj.to_dict(),
                    collection=self._ref_task_step_collection,
                )

            self._kvstore.delete(ref_task_step_id, collection=self._metadata_collection)

    def delete_ref_task_step(self, ref_task_step_id: str, raise_error: bool = True) -> None:
        """Delete a ref_task_step and all it's associated nodes."""
        ref_task_step_info = self.get_ref_task_step_info(ref_task_step_id)
        if ref_task_step_info is None:
            if raise_error:
                raise ValueError(f"ref_task_step_id {ref_task_step_id} not found.")
            else:
                return

        for task_step_id in ref_task_step_info.node_ids:
            self.delete_task_step(task_step_id, raise_error=False, remove_ref_task_step_node=False)

        self._kvstore.delete(ref_task_step_id, collection=self._metadata_collection)
        self._kvstore.delete(ref_task_step_id, collection=self._ref_task_step_collection)
