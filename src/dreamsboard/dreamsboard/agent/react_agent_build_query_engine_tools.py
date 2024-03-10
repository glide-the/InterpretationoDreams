from typing import List

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage, Document,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def build_query_docs() -> list[list[Document]]:
    # load data
    march_docs = SimpleDirectoryReader(
        input_files=["/home/dmeck/Documents/张毛峰个人简历 - 2024-01-20(1).pdf"]
    ).load_data()
    june_docs = SimpleDirectoryReader(
        input_files=["/home/dmeck/Downloads/个人简历_刘立兼(1).docx"]
    ).load_data()
    sept_docs = SimpleDirectoryReader(
        input_files=["/home/dmeck/Downloads/简历_宋金珂_北京交通大学_网络空间安全.pdf"]
    ).load_data()
    return [march_docs, june_docs, sept_docs]


def build_query_docs_index_store(query_docs: list[list[Document]], embeddings: OpenAIEmbedding) -> list[VectorStoreIndex]:
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/marchtest2"
        )
        march_index = load_index_from_storage(storage_context)
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/junetest2"
        )
        june_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/septtest2"
        )
        sept_index = load_index_from_storage(storage_context)
        index_loaded = True
    except:
        index_loaded = False
    if not index_loaded:
        # load data
        march_docs = query_docs[0]
        june_docs = query_docs[1]
        sept_docs = query_docs[2]
        # build index
        march_index = VectorStoreIndex.from_documents(
            march_docs, embed_model=embeddings
        )
        june_index = VectorStoreIndex.from_documents(
            june_docs, embed_model=embeddings
        )
        sept_index = VectorStoreIndex.from_documents(
            sept_docs, embed_model=embeddings
        )

        # persist index
        march_index.storage_context.persist(persist_dir="./storage/marchtest2")

        june_index.storage_context.persist(persist_dir="./storage/junetest2")
        sept_index.storage_context.persist(persist_dir="./storage/septtest2")

    index_store = [march_index, june_index, sept_index]
    return index_store


def build_query_engine_tools(llm: OpenAI, index_store: list[VectorStoreIndex]) -> list[QueryEngineTool]:
    march_index = index_store[0]
    june_index = index_store[1]
    sept_index = index_store[2]
    march_engine = march_index.as_query_engine(similarity_top_k=3, llm=llm)
    june_engine = june_index.as_query_engine(similarity_top_k=3, llm=llm)
    sept_engine = sept_index.as_query_engine(similarity_top_k=3, llm=llm)

    query_tool_march = QueryEngineTool.from_defaults(
        query_engine=march_engine,
        name="resume_zhangmaofeng",
        description=(
            f"关于张毛峰的简历信息，包括了langchain-chatchat、InterpretationoDreams、KM 平台、省检修特高压生产指挥管控系统、智能运检移动应用、福建监控系统项目"
            f"eg:检索时带上详细的问题内容"

        ),
    )

    query_tool_june = QueryEngineTool.from_defaults(
        query_engine=june_engine,
        name="resume_liulijian",
        description=(
            f"关于刘立兼的简历信息，包括了•篝火心理、雷鸟365、雷鸟365、网聚宝CRM、AP数据基盘、AP数据基盘等项目"
            f"eg:检索时带上详细的问题内容"

        ),
    )
    query_tool_sept = QueryEngineTool.from_defaults(
        query_engine=sept_engine,
        name="resume_songjinke",
        description=(
            f"关于宋金珂的简历信息，包括了全球 IPv4 空间内的物联网设备扫描识别和隐私安全分、开源软件生态内的跨项目依赖分析及漏洞影响追溯、已发表论文列表、IoT 设备安全等项目"
            f"eg:检索时带上详细的问题内容"

        ),
    )

    query_engine_tools = [query_tool_march, query_tool_june, query_tool_sept]

    return query_engine_tools
