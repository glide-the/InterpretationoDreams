import os

from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from dreamsboard.vector.knowledge_base.kb_cache.base import *


# patch FAISS to include doc id in Document.metadata
def _new_ds_search(self, search: str) -> Union[str, Document]:
    if search not in self._dict:
        return f"ID {search} not found."
    else:
        doc = self._dict[search]
        if isinstance(doc, Document):
            doc.metadata["id"] = search
        return doc


InMemoryDocstore.search = _new_ds_search


class ThreadSafeFaiss(ThreadSafeObject):
    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"

    def docs_count(self) -> int:
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
        with self.acquire():
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            ret = self._obj.save_local(path)
            logger.info(f"已将向量库 {self.key} 保存到磁盘")
        return ret

    def clear(self):
        ret = []
        with self.acquire():
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"已将向量库 {self.key} 清空")
        return ret


class _FaissPool(CachePool):
    def new_vector_store(
        self,
        embed_model: str,
        device: str = "cpu",
    ) -> FAISS:
        # create an empty vector store
        model_kwargs = {"device": device}
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs=model_kwargs,
            show_progress=True,
        )
        doc = Document(page_content="init", metadata={})
        vector_store = FAISS.from_documents([doc], embeddings, normalize_L2=True)
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def save_vector_store(self, kb_name: str, path: str = None):
        if cache := self.get(kb_name):
            return cache.save(path)

    def unload_vector_store(self, kb_name: str):
        if cache := self.get(kb_name):
            self.pop(kb_name)
            logger.info(f"成功释放向量库：{kb_name}")


class KBFaissPool(_FaissPool):
    def load_vector_store(
        self,
        kb_name: str,
        embed_model: str,
        vector_name: str = None,
        create: bool = True,
        device: str = "cpu",
    ) -> ThreadSafeFaiss:
        self.atomic.acquire()
        locked = True
        vector_name = vector_name or embed_model.replace(":", "_")
        cache = self.get((kb_name, vector_name))  # 用元组比拼接字符串好一些
        try:
            if cache is None:
                item = ThreadSafeFaiss((kb_name, vector_name), pool=self)
                self.set((kb_name, vector_name), item)
                with item.acquire(msg="初始化"):
                    self.atomic.release()
                    locked = False
                    logger.info(
                        f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk."
                    )
                    vs_path = os.path.join(kb_name, "vector_store", vector_name)

                    if os.path.isfile(os.path.join(vs_path, "index.faiss")):
                        model_kwargs = {"device": device}
                        embeddings = HuggingFaceEmbeddings(
                            model_name=embed_model,
                            model_kwargs=model_kwargs,
                            show_progress=True,
                        )
                        vector_store = FAISS.load_local(
                            vs_path,
                            embeddings,
                            normalize_L2=True,
                            allow_dangerous_deserialization=True,
                        )
                    elif create:
                        # create an empty vector store
                        if not os.path.exists(vs_path):
                            os.makedirs(vs_path)
                        vector_store = self.new_vector_store(
                            embed_model=embed_model, device=device
                        )
                        vector_store.save_local(vs_path)
                    else:
                        raise RuntimeError(f"knowledge base {kb_name} not exist.")
                    item.obj = vector_store
                    item.finish_loading()
            else:
                self.atomic.release()
                locked = False
        except Exception as e:
            if locked:  # we don't know exception raised before or after atomic.release
                self.atomic.release()
            logger.exception(e)
            raise RuntimeError(f"向量库 {kb_name} 加载失败。")
        return self.get((kb_name, vector_name))


kb_faiss_pool = KBFaissPool(cache_num=1)
