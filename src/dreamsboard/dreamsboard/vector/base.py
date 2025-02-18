import os
import shutil
from typing import Dict, List, Tuple

from langchain.docstore.document import Document


class DocumentWithVSId(Document):
    """
    矢量化后的文档
    """

    id: str = None


class CollectionService:
    _vs_path: str
    device: str
    kb_name: str
    embed_model: str
    vector_name: str = None

    def __init__(
        self, kb_name: str, embed_model: str, vector_name: str, device: str = "cpu"
    ) -> None:
        self.device = device
        self.kb_name = kb_name
        self.embed_model = embed_model
        self.vector_name = vector_name
        self._vs_path = os.path.join(kb_name, "vector_store", vector_name)

    def save_vector_store(self):
        raise NotImplementedError

    def get_doc_by_ids(self, ids: List[str]) -> List[DocumentWithVSId]:
        raise NotImplementedError

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        raise NotImplementedError

    def do_create_kb(self):
        raise NotImplementedError

    def do_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float = 1,
    ) -> List[DocumentWithVSId]:
        raise NotImplementedError

    def do_add_doc(
        self,
        docs: List[DocumentWithVSId],
        **kwargs,
    ) -> List[Dict]:
        raise NotImplementedError

    def do_clear_vs(self):
        raise NotImplementedError
