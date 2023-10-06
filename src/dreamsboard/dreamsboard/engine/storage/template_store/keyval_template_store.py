"""Template store."""

from typing import Dict, Optional, Sequence

from dreamsboard.engine.schema import BaseNode
from dreamsboard.engine.generate.code_generate import (
    CodeGenerator
)

from dreamsboard.engine.storage.template_store.types import BaseTemplateStore, RefTemplateInfo
from dreamsboard.engine.storage.template_store.utils import template_to_json, json_to_template
from dreamsboard.engine.storage.kvstore.types import BaseKVStore

DEFAULT_NAMESPACE = "template_store"


class KVTemplateStore(BaseTemplateStore):
    """Template (Node) store.

    NOTE: at the moment, this store is primarily used to store Node objects.
    Each node will be assigned an ID.

    The same templatestore can be reused across index structures. This
    allows you to reuse the same storage for multiple index structures;
    otherwise, each index would create a templatestore under the hood.

    .. code-block:: python
        nodes = SimpleNodeParser.get_nodes_from_templates()
        templatestore = SimpleTemplateStore()
        templatestore.add_templates(nodes)
        storage_context = StorageContext.from_defaults(templatestore=templatestore)

        summary_index = SummaryIndex(nodes, storage_context=storage_context)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        keyword_table_index = SimpleKeywordTableIndex(
            nodes,
            storage_context=storage_context
        )

    This will use the same templatestore for multiple index structures.

    Args:
        kvstore (BaseKVStore): key-value store
        namespace (str): namespace for the templatestore

    """

    def __init__(
        self,
        kvstore: BaseKVStore,
        namespace: Optional[str] = None,
    ) -> None:
        """Init a KVTemplateStore."""
        self._kvstore = kvstore
        self._namespace = namespace or DEFAULT_NAMESPACE
        self._node_collection = f"{self._namespace}/data"
        self._ref_template_collection = f"{self._namespace}/ref_template_info"
        self._metadata_collection = f"{self._namespace}/metadata"

    @property
    def templates(self) -> Dict[str, BaseNode]:
        """Get all templates.

        Returns:
            Dict[str, BaseTemplate]: templates

        """
        json_dict = self._kvstore.get_all(collection=self._node_collection)
        return {key: json_to_template(json) for key, json in json_dict.items()}

    def add_templates(
        self, nodes: Sequence[BaseNode], allow_update: bool = True
    ) -> None:
        """Add a template to the store.

        Args:
            templates (List[BaseTemplate]): templates
            allow_update (bool): allow update of templatestore from template
            :param allow_update:
            :param nodes:

        """
        for node in nodes:
            # NOTE: template could already exist in the store, but we overwrite it
            if not allow_update and self.template_exists(node.node_id):
                raise ValueError(
                    f"node_id {node.node_id} already exists. "
                    "Set allow_update to True to overwrite."
                )
            node_key = node.node_id
            data = template_to_json(node)
            self._kvstore.put(node_key, data, collection=self._node_collection)

            # update template_collection if needed
            metadata = {"template_hash": node.hash}
            if isinstance(node, CodeGenerator) and node.ref_template_id is not None:
                ref_template_info = self.get_ref_template_info(node.ref_template_id) or RefTemplateInfo()
                ref_template_info.node_ids.append(node.node_id)
                if not ref_template_info.metadata:
                    ref_template_info.metadata = node.metadata or {}
                self._kvstore.put(
                    node.ref_template_id,
                    ref_template_info.to_dict(),
                    collection=self._ref_template_collection,
                )

                # update metadata with map
                metadata["ref_template_id"] = node.ref_template_id
                self._kvstore.put(
                    node_key, metadata, collection=self._metadata_collection
                )
            else:
                self._kvstore.put(
                    node_key, metadata, collection=self._metadata_collection
                )

    def get_template(self, template_id: str, raise_error: bool = True) -> Optional[BaseNode]:
        """Get a template from the store.

        Args:
            template_id (str): template id
            raise_error (bool): raise error if template_id not found

        """
        json = self._kvstore.get(template_id, collection=self._node_collection)
        if json is None:
            if raise_error:
                raise ValueError(f"template_id {template_id} not found.")
            else:
                return None
        return json_to_template(json)

    def get_ref_template_info(self, ref_template_id: str) -> Optional[RefTemplateInfo]:
        """Get the RefTemplateInfo for a given ref_template_id."""
        ref_template_info = self._kvstore.get(
            ref_template_id, collection=self._ref_template_collection
        )
        if not ref_template_info:
            return None

        # TODO: deprecated legacy support
        if "template_ids" in ref_template_info:
            ref_template_info["node_ids"] = ref_template_info.get("template_ids", [])
            ref_template_info.pop("template_ids")

            ref_template_info["metadata"] = ref_template_info.get("extra_info", {})
            ref_template_info.pop("extra_info")

        return RefTemplateInfo(**ref_template_info)

    def get_all_ref_template_info(self) -> Optional[Dict[str, RefTemplateInfo]]:
        """Get a mapping of ref_template_id -> RefTemplateInfo for all ingested templates."""
        ref_template_infos = self._kvstore.get_all(collection=self._ref_template_collection)
        if ref_template_infos is None:
            return None

        # TODO: deprecated legacy support
        all_ref_template_infos = {}
        for template_id, ref_template_info in ref_template_infos.items():
            if "template_ids" in ref_template_info:
                ref_template_info["node_ids"] = ref_template_info.get("template_ids", [])
                ref_template_info.pop("template_ids")

                ref_template_info["metadata"] = ref_template_info.get("extra_info", {})
                ref_template_info.pop("extra_info")
                all_ref_template_infos[template_id] = RefTemplateInfo(**ref_template_info)

        return all_ref_template_infos

    def ref_template_exists(self, ref_template_id: str) -> bool:
        """Check if a ref_template_id has been ingested."""
        return self.get_ref_template_info(ref_template_id) is not None

    def template_exists(self, template_id: str) -> bool:
        """Check if template exists."""
        return self._kvstore.get(template_id, self._node_collection) is not None

    def _remove_ref_template_node(self, template_id: str) -> None:
        """Helper function to remove node template_id from ref_template_collection."""
        metadata = self._kvstore.get(template_id, collection=self._metadata_collection)
        if metadata is None:
            return

        ref_template_id = metadata.get("ref_template_id", None)

        if ref_template_id is None:
            return

        ref_template_info = self._kvstore.get(
            ref_template_id, collection=self._ref_template_collection
        )

        if ref_template_info is not None:
            ref_template_obj = RefTemplateInfo(**ref_template_info)

            ref_template_obj.node_ids.remove(template_id)

            # delete ref_template from collection if it has no more template_ids
            if len(ref_template_obj.node_ids) > 0:
                self._kvstore.put(
                    ref_template_id,
                    ref_template_obj.to_dict(),
                    collection=self._ref_template_collection,
                )

            self._kvstore.delete(ref_template_id, collection=self._metadata_collection)

    def delete_template(
        self, template_id: str, raise_error: bool = True, remove_ref_template_node: bool = True
    ) -> None:
        """Delete a template from the store."""
        if remove_ref_template_node:
            self._remove_ref_template_node(template_id)

        delete_success = self._kvstore.delete(template_id, collection=self._node_collection)
        _ = self._kvstore.delete(template_id, collection=self._metadata_collection)

        if not delete_success and raise_error:
            raise ValueError(f"template_id {template_id} not found.")

    def delete_ref_template(self, ref_template_id: str, raise_error: bool = True) -> None:
        """Delete a ref_template and all it's associated nodes."""
        ref_template_info = self.get_ref_template_info(ref_template_id)
        if ref_template_info is None:
            if raise_error:
                raise ValueError(f"ref_template_id {ref_template_id} not found.")
            else:
                return

        for template_id in ref_template_info.node_ids:
            self.delete_template(template_id, raise_error=False, remove_ref_template_node=False)

        self._kvstore.delete(ref_template_id, collection=self._metadata_collection)
        self._kvstore.delete(ref_template_id, collection=self._ref_template_collection)

    def set_template_hash(self, template_id: str, template_hash: str) -> None:
        """Set the hash for a given template_id."""
        metadata = {"template_hash": template_hash}
        self._kvstore.put(template_id, metadata, collection=self._metadata_collection)

    def get_template_hash(self, template_id: str) -> Optional[str]:
        """Get the stored hash for a template, if it exists."""
        metadata = self._kvstore.get(template_id, collection=self._metadata_collection)
        if metadata is not None:
            return metadata.get("template_hash", None)
        else:
            return None
