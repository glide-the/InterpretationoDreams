import os
import shutil
from typing import Any, Dict, List, Sized, Tuple

from langchain.docstore.document import Document

from dreamsboard.vector.base import CollectionService, DocumentWithVSId
from dreamsboard.vector.knowledge_base.kb_cache.faiss_cache import (
    ThreadSafeFaiss,
    kb_faiss_pool,
)


def _len_check_if_sized(x: Any, y: Any, x_name: str, y_name: str) -> None:
    if isinstance(x, Sized) and isinstance(y, Sized) and len(x) != len(y):
        raise ValueError(
            f"{x_name} and {y_name} expected to be equal length but "
            f"len({x_name})={len(x)} and len({y_name})={len(y)}"
        )
    return


class FaissCollectionService(CollectionService):
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
        self.load_vector_store()

    def load_vector_store(self) -> ThreadSafeFaiss:
        return kb_faiss_pool.load_vector_store(
            kb_name=self.kb_name,
            vector_name=self.vector_name,
            embed_model=self.embed_model,
            device=self.device,
        )

    def save_vector_store(self):
        self.load_vector_store().save(self._vs_path)

    def get_doc_by_ids(self, ids: List[str]) -> List[DocumentWithVSId]:
        if not isinstance(ids[0], str):
            raise ValueError(f"ids expected to be List[str] but got {type(ids)}")

        with self.load_vector_store().acquire() as vs:
            docs = []
            for id in ids:
                store_data = vs.docstore._dict.get(id)
                if store_data is None:
                    continue
                doc = DocumentWithVSId(**{**store_data.dict(), "id": id})
                docs.append(doc)
            return docs

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        with self.load_vector_store().acquire() as vs:
            vs.delete(ids)

    def do_create_kb(self):
        if not os.path.exists(self._vs_path):
            os.makedirs(self._vs_path)
        self.load_vector_store()

    def do_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float = 1,
    ) -> List[DocumentWithVSId]:
        with self.load_vector_store().acquire(msg="查询") as vs:
            faiss_retriever = vs.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score_threshold, "k": top_k},
            )
            docs = faiss_retriever.invoke(query)
            docs = [
                DocumentWithVSId(**{**doc.dict(), "id": doc.metadata["id"]})
                for doc in docs
            ]
        return docs

    def do_add_doc(
        self,
        docs: List[DocumentWithVSId],
        **kwargs,
    ) -> List[Dict]:
        if docs is None or len(docs) == 0:
            return []
        if not isinstance(docs[0], DocumentWithVSId):
            raise ValueError(
                f"docs expected to be List[DocumentWithVSId] but got {type(docs)}"
            )

        ids = [x.id for x in docs]
        texts = [x.page_content for x in docs]
        metadatas = [x.metadata for x in docs]

        _len_check_if_sized(docs, ids, "docs", "ids")
        _len_check_if_sized(docs, texts, "docs", "texts")
        _len_check_if_sized(docs, metadatas, "docs", "metadatas")
        with self.load_vector_store().acquire(msg="插入") as vs:
            embeddings = vs.embeddings.embed_documents(texts)
            ids = vs.add_embeddings(
                text_embeddings=zip(texts, embeddings), metadatas=metadatas, ids=ids
            )

        self.save_vector_store()
        doc_infos = [
            DocumentWithVSId(**{**doc.dict(), "id": id}) for id, doc in zip(ids, docs)
        ]
        return doc_infos

    def do_clear_vs(self):
        with kb_faiss_pool.atomic:
            kb_faiss_pool.pop((self.kb_name, self.vector_name))
        try:
            shutil.rmtree(self._vs_path)
        except Exception:
            ...
        os.makedirs(self._vs_path, exist_ok=True)
