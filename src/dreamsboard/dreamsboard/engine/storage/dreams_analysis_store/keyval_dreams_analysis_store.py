"""analysis store."""

from typing import Dict, Optional, Sequence

from dreamsboard.engine.entity.dreams_personality.dreams_personality import DreamsPersonalityNode
from dreamsboard.engine.generate.code_generate import (
    CodeGenerator
)
from dreamsboard.engine.storage.dreams_analysis_store.types import BaseDreamsAnalysisStore, RefAnalysisInfo
from dreamsboard.engine.storage.dreams_analysis_store.utils import json_to_analysis, analysis_to_json

from dreamsboard.engine.storage.kvstore.types import BaseKVStore

DEFAULT_NAMESPACE = "analysis_store"


class KVDreamsAnalysisStore(BaseDreamsAnalysisStore):
    """analysis (Node) store.

    NOTE: at the moment, this store is primarily used to store Node objects.
    Each node will be assigned an ID.

    The same analysisstore can be reused across index structures. This
    allows you to reuse the same storage for multiple index structures;
    otherwise, each index would create a analysisstore under the hood.

    .. code-block:: python
            store = SimpleDreamsAnalysisStore()
            dreams = DreamsPersonalityNode.from_config(cfg={
                "dreams_guidance_context": "dreams_guidance",
                "dreams_personality_context": "dreams_personality"
            })
            store.add_analysis([dreams])
            logger.info(store.analysis_all)
            dreams_analysis_store_path = concat_dirs(dirname="./storage", basename="dreams_analysis_store.json")
            store.persist(persist_path=dreams_analysis_store_path)

    This will use the same analysisstore for multiple index structures.

    Args:
        kvstore (BaseKVStore): key-value store
        namespace (str): namespace for the analysisstore

    """

    def __init__(
            self,
            kvstore: BaseKVStore,
            namespace: Optional[str] = None,
    ) -> None:
        """Init a KVanalysisStore."""
        self._kvstore = kvstore
        self._namespace = namespace or DEFAULT_NAMESPACE
        self._node_collection = f"{self._namespace}/data"
        self._ref_analysis_collection = f"{self._namespace}/ref_analysis_info"
        self._metadata_collection = f"{self._namespace}/metadata"

    @property
    def analysis_all(self) -> Dict[str, DreamsPersonalityNode]:
        """Get all analysis.

        Returns:
            Dict[str, DreamsPersonalityNode]: analysis

        """
        json_dict = self._kvstore.get_all(collection=self._node_collection)
        return {key: json_to_analysis(json) for key, json in json_dict.items()}

    def add_analysis(
            self, analyses: Sequence[DreamsPersonalityNode], allow_update: bool = True
    ) -> None:
        """Add a analyses to the store.

        Args:
            analyses (List[DreamsPersonalityNode]): analyses
            allow_update (bool): allow update of store from analyses
            :param allow_update:
            :param analyses:

        """
        for node in analyses:
            # NOTE: analysis could already exist in the store, but we overwrite it
            if not allow_update and self.analysis_exists(node.node_id):
                raise ValueError(
                    f"node_id {node.node_id} already exists. "
                    "Set allow_update to True to overwrite."
                )
            node_key = node.node_id
            data = analysis_to_json(node)
            self._kvstore.put(node_key, data, collection=self._node_collection)

            # update analysis_collection if needed
            metadata = {"analysis_hash": node.hash}
            if isinstance(node, DreamsPersonalityNode) and node.ref_analysis_id is not None:
                ref_analysis_info = self.get_ref_analysis_info(node.ref_analysis_id) or RefAnalysisInfo()
                ref_analysis_info.node_ids.append(node.node_id)
                if not ref_analysis_info.metadata:
                    ref_analysis_info.metadata = node.metadata or {}
                self._kvstore.put(
                    node.ref_analysis_id,
                    ref_analysis_info.to_dict(),
                    collection=self._ref_analysis_collection,
                )

                # update metadata with map
                metadata["ref_analysis_id"] = node.ref_analysis_id
                self._kvstore.put(
                    node_key, metadata, collection=self._metadata_collection
                )
            else:
                self._kvstore.put(
                    node_key, metadata, collection=self._metadata_collection
                )

    def get_analysis(self, analysis_id: str, raise_error: bool = True) -> Optional[DreamsPersonalityNode]:
        """Get a analysis from the store.

        Args:
            analysis_id (str): analysis id
            raise_error (bool): raise error if analysis_id not found

        """
        json = self._kvstore.get(analysis_id, collection=self._node_collection)
        if json is None:
            if raise_error:
                raise ValueError(f"analysis_id {analysis_id} not found.")
            else:
                return None
        return json_to_analysis(json)

    def delete_analysis(
            self, analysis_id: str, raise_error: bool = True, remove_ref_analysis_node: bool = True
    ) -> None:
        """Delete a analysis from the store."""
        if remove_ref_analysis_node:
            self._remove_ref_analysis_node(analysis_id)

        delete_success = self._kvstore.delete(analysis_id, collection=self._node_collection)
        _ = self._kvstore.delete(analysis_id, collection=self._metadata_collection)

        if not delete_success and raise_error:
            raise ValueError(f"analysis_id {analysis_id} not found.")

    def analysis_exists(self, analysis_id: str) -> bool:
        """Check if analysis exists."""
        return self._kvstore.get(analysis_id, self._node_collection) is not None

    def get_ref_analysis_info(self, ref_analysis_id: str) -> Optional[RefAnalysisInfo]:
        """Get the RefAnalysisInfo for a given ref_analysis_id."""
        ref_analysis_info = self._kvstore.get(
            ref_analysis_id, collection=self._ref_analysis_collection
        )
        if not ref_analysis_info:
            return None

        # TODO: deprecated legacy support
        if "analysis_ids" in ref_analysis_info:
            ref_analysis_info["node_ids"] = ref_analysis_info.get("analysis_ids", [])
            ref_analysis_info.pop("analysis_ids")

            ref_analysis_info["metadata"] = ref_analysis_info.get("extra_info", {})
            ref_analysis_info.pop("extra_info")

        return RefAnalysisInfo(**ref_analysis_info)

    def get_all_ref_analysis_info(self) -> Optional[Dict[str, RefAnalysisInfo]]:
        """Get a mapping of ref_analysis_id -> RefAnalysisInfo for all ingested nodes."""
        ref_analysis_infos = self._kvstore.get_all(collection=self._ref_analysis_collection)
        if ref_analysis_infos is None:
            return None

        # TODO: deprecated legacy support
        all_ref_analysis_infos = {}
        for analysis_id, ref_analysis_info in ref_analysis_infos.items():
            if "analysis_ids" in ref_analysis_info:
                ref_analysis_info["node_ids"] = ref_analysis_info.get("analysis_ids", [])
                ref_analysis_info.pop("analysis_ids")

                ref_analysis_info["metadata"] = ref_analysis_info.get("extra_info", {})
                ref_analysis_info.pop("extra_info")
                all_ref_analysis_infos[analysis_id] = RefAnalysisInfo(**ref_analysis_info)

        return all_ref_analysis_infos

    def ref_analysis_exists(self, ref_analysis_id: str) -> bool:
        """Check if a ref_analysis_id has been ingested."""
        return self.get_ref_analysis_info(ref_analysis_id) is not None

    def _remove_ref_analysis_node(self, analysis_id: str) -> None:
        """Helper function to remove node analysis_id from ref_analysis_collection."""
        metadata = self._kvstore.get(analysis_id, collection=self._metadata_collection)
        if metadata is None:
            return

        ref_analysis_id = metadata.get("ref_analysis_id", None)

        if ref_analysis_id is None:
            return

        ref_analysis_info = self._kvstore.get(
            ref_analysis_id, collection=self._ref_analysis_collection
        )

        if ref_analysis_info is not None:
            ref_analysis_obj = RefAnalysisInfo(**ref_analysis_info)

            ref_analysis_obj.node_ids.remove(analysis_id)

            # delete ref_analysis from collection if it has no more analysis_ids
            if len(ref_analysis_obj.node_ids) > 0:
                self._kvstore.put(
                    ref_analysis_id,
                    ref_analysis_obj.to_dict(),
                    collection=self._ref_analysis_collection,
                )

            self._kvstore.delete(ref_analysis_id, collection=self._metadata_collection)

    def delete_ref_analysis(self, ref_analysis_id: str, raise_error: bool = True) -> None:
        """Delete a ref_analysis and all it's associated nodes."""
        ref_analysis_info = self.get_ref_analysis_info(ref_analysis_id)
        if ref_analysis_info is None:
            if raise_error:
                raise ValueError(f"ref_analysis_id {ref_analysis_id} not found.")
            else:
                return

        for analysis_id in ref_analysis_info.node_ids:
            self.delete_analysis(analysis_id, raise_error=False, remove_ref_analysis_node=False)

        self._kvstore.delete(ref_analysis_id, collection=self._metadata_collection)
        self._kvstore.delete(ref_analysis_id, collection=self._ref_analysis_collection)
