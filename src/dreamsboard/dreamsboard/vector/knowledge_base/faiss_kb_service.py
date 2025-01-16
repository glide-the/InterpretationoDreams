import os
import shutil
from typing import Dict, List, Tuple

from langchain.docstore.document import Document
 
from dreamsboard.vector.knowledge_base.kb_cache.faiss_cache import (
    ThreadSafeFaiss,
    kb_faiss_pool,
) 


class FaissKBService:
    _vs_path: str
    device: str
    kb_name: str
    embed_model: str
    vector_name: str = None

    def __init__(self, kb_name: str, embed_model: str, vector_name: str, device: str = 'cpu') -> None:
        self.device = device
        self.kb_name = kb_name
        self.embed_model = embed_model
        self.vector_name = vector_name
        self._vs_path = os.path.join(kb_name, "vector_store", vector_name)
 
    def load_vector_store(self) -> ThreadSafeFaiss:
        return kb_faiss_pool.load_vector_store(
            kb_name=self.kb_name,
            vector_name=self.vector_name,
            embed_model=self.embed_model,
            device=self.device,
        )

    def save_vector_store(self):
        self.load_vector_store().save(self._vs_path)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        with self.load_vector_store().acquire() as vs:
            return [vs.docstore._dict.get(id) for id in ids]

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
    ) -> List[Tuple[Document, float]]:
        with self.load_vector_store().acquire() as vs:
            faiss_retriever = vs.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score_threshold, "k": top_k},
            )
            docs = faiss_retriever.invoke(query)
        return docs

    def do_add_doc(
        self,
        docs: List[Document],
        **kwargs,
    ) -> List[Dict]:
        texts = [x.page_content for x in docs]
        metadatas = [x.metadata for x in docs]
        with self.load_vector_store().acquire() as vs:
            embeddings = vs.embeddings.embed_documents(texts)
            ids = vs.add_embeddings(
                text_embeddings=zip(texts, embeddings), metadatas=metadatas
            )
            if not kwargs.get("not_refresh_vs_cache"):
                vs.save_local(self._vs_path)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos


    def do_clear_vs(self):
        with kb_faiss_pool.atomic:
            kb_faiss_pool.pop((self.kb_name, self.vector_name))
        try:
            shutil.rmtree(self._vs_path)
        except Exception:
            ...
        os.makedirs(self._vs_path, exist_ok=True)

 

 
