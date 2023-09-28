from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
)
from llama_index.llms import OpenAI

from llama_index.tools import QueryEngineTool, ToolMetadata
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
# llm = OpenAI(temperature=0, model="gpt-4-0613")
service_context = ServiceContext.from_defaults(llm=llm)



try:
    # load index from disk
    """
    
    # 从磁盘加载索引，索引是一个文件夹，里面有很多文件，文件内容由索引构建器、矢量存储器、文档存储器构成
    docstore: BaseDocumentStore
    index_store: BaseIndexStore
    vector_store: VectorStore
    graph_store: GraphStore
    
    # BaseIndexStore的索引实现由KVIndexStore实现，它的作用是将文档转换为索引结构，然后存储到KVStore中
    # KVIndexStore的子类有SimpleIndexStore，RedisIndexStore，MongoIndexStore
    # 这是一个缓存模块，用于存储索引结构，以便下次使用，避免每次文件加载，都要读取磁盘
    """
    storage_context = StorageContext.from_defaults(persist_dir="./storage/march")
    march_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(persist_dir="./storage/june")
    june_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(persist_dir="./storage/sept")
    sept_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False


if not index_loaded:
    # load data
    march_docs = SimpleDirectoryReader(
        input_files=["../../data/10q/uber_10q_march_2022.pdf"]
    ).load_data()
    june_docs = SimpleDirectoryReader(
        input_files=["../../data/10q/uber_10q_june_2022.pdf"]
    ).load_data()
    sept_docs = SimpleDirectoryReader(
        input_files=["../../data/10q/uber_10q_sept_2022.pdf"]
    ).load_data()

    # build index
    march_index = VectorStoreIndex.from_documents(
        march_docs, service_context=service_context
    )
    june_index = VectorStoreIndex.from_documents(
        june_docs, service_context=service_context
    )
    sept_index = VectorStoreIndex.from_documents(
        sept_docs, service_context=service_context
    )

    # persist index to disk
    march_index.storage_context.persist(persist_dir="./storage/march")
    june_index.storage_context.persist(persist_dir="./storage/june")
    sept_index.storage_context.persist(persist_dir="./storage/sept")


# 构建查询引擎，引擎可以被拓展为各种查询工具，比如：搜索引擎、问答引擎、推荐引擎
# 同样索引构造BaseIndex也可以构建问答引擎，问答引擎包含由llama_index.chat_engine.types.ChatModel定义的问答模型
# RetrieverQueryEngine提供了一个基于检索的问答方法_query，** retrieve是VectorIndexRetriever的方法，用于检索文档**
"""
 def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
     
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self.retrieve(query_bundle)

                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )

            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
"""
march_engine = march_index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)
june_engine = june_index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)
sept_engine = sept_index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)
from llama_index.tools.query_engine import QueryEngineTool


# 包装查询引擎为调用工具
query_tool_sept = QueryEngineTool.from_defaults(
    query_engine=sept_engine,
    name="sept_2022",
    description=f"Provides information about Uber quarterly financials ending September 2022",
)
