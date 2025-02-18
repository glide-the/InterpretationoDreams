from typing import List

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def build_query_docs() -> list[list[Document]]:
    # load data

    # load data
    meas_docs = SimpleDirectoryReader(
        input_files=[
            "/home/dmeck/Documents/电网业务中台/电网资源业务中台接口开发规范说明书V2.01-测点管理中心.docx"
        ]
    ).load_data()
    topo_docs = SimpleDirectoryReader(
        input_files=[
            "/home/dmeck/Documents/电网业务中台/电网资源业务中台接口开发规范说明书V2.01-电网拓扑中心.docx"
        ]
    ).load_data()
    geo_docs = SimpleDirectoryReader(
        input_files=[
            "/home/dmeck/Documents/电网业务中台/电网资源业务中台接口开发规范说明书V2.01-电网图形中心.docx"
        ]
    ).load_data()
    asset_docs = SimpleDirectoryReader(
        input_files=[
            "/home/dmeck/Documents/电网业务中台/电网资源业务中台接口开发规范说明书V2.01-电网资产中心.docx"
        ]
    ).load_data()
    psr_docs = SimpleDirectoryReader(
        input_files=[
            "/home/dmeck/Documents/电网业务中台/电网资源业务中台接口开发规范说明书V2.01-电网资源中心.docx"
        ]
    ).load_data()
    base_docs = SimpleDirectoryReader(
        input_files=[
            "/home/dmeck/Documents/电网业务中台/电网资源业务中台接口开发规范说明书V2.01-基础服务中心.docx"
        ]
    ).load_data()
    return [meas_docs, topo_docs, geo_docs, asset_docs, psr_docs, base_docs]


def build_query_docs_index_store(
    query_docs: list[list[Document]], embeddings: OpenAIEmbedding
) -> list[VectorStoreIndex]:
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage/meas")
        meas_index = load_index_from_storage(storage_context)
        storage_context = StorageContext.from_defaults(persist_dir="./storage/topo")
        topo_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(persist_dir="./storage/geo")
        geo_index = load_index_from_storage(storage_context)
        storage_context = StorageContext.from_defaults(persist_dir="./storage/asset")
        asset_index = load_index_from_storage(storage_context)
        storage_context = StorageContext.from_defaults(persist_dir="./storage/psr")
        psr_index = load_index_from_storage(storage_context)
        storage_context = StorageContext.from_defaults(persist_dir="./storage/base")
        base_index = load_index_from_storage(storage_context)
        index_loaded = True
    except:
        index_loaded = False
    if not index_loaded:
        # load data
        meas_docs = query_docs[0]
        topo_docs = query_docs[1]
        geo_docs = query_docs[2]
        asset_docs = query_docs[3]
        psr_docs = query_docs[4]
        base_docs = query_docs[5]

        # build index
        meas_index = VectorStoreIndex.from_documents(meas_docs, embed_model=embeddings)
        topo_index = VectorStoreIndex.from_documents(topo_docs, embed_model=embeddings)
        geo_index = VectorStoreIndex.from_documents(geo_docs, embed_model=embeddings)
        asset_index = VectorStoreIndex.from_documents(
            asset_docs, embed_model=embeddings
        )
        psr_index = VectorStoreIndex.from_documents(psr_docs, embed_model=embeddings)
        base_index = VectorStoreIndex.from_documents(base_docs, embed_model=embeddings)

        # persist index
        meas_index.storage_context.persist(persist_dir="./storage/meas")
        topo_index.storage_context.persist(persist_dir="./storage/topo")
        geo_index.storage_context.persist(persist_dir="./storage/geo")
        asset_index.storage_context.persist(persist_dir="./storage/asset")
        psr_index.storage_context.persist(persist_dir="./storage/psr")
        base_index.storage_context.persist(persist_dir="./storage/base")

    index_store = [
        meas_index,
        topo_index,
        geo_index,
        asset_index,
        psr_index,
        base_index,
    ]
    return index_store


def build_query_engine_tools(
    llm: OpenAI, index_store: list[VectorStoreIndex]
) -> list[QueryEngineTool]:
    meas_index = index_store[0]
    topo_index = index_store[1]
    geo_index = index_store[2]
    asset_index = index_store[3]
    psr_index = index_store[4]
    base_index = index_store[5]
    meas_engine = meas_index.as_query_engine(similarity_top_k=3, llm=llm)
    topo_engine = topo_index.as_query_engine(similarity_top_k=3, llm=llm)
    geo_engine = geo_index.as_query_engine(similarity_top_k=3, llm=llm)
    asset_engine = asset_index.as_query_engine(similarity_top_k=3, llm=llm)
    psr_engine = psr_index.as_query_engine(similarity_top_k=3, llm=llm)
    base_engine = base_index.as_query_engine(similarity_top_k=3, llm=llm)
    query_tool_meas = QueryEngineTool.from_defaults(
        query_engine=meas_engine,
        name="meas",
        description=(
            f"关于测点管理中心的业务中台，包括了测量查询服务、事件中心服务、事件类型定义、事件代码表、策略类型、实时开源状态 等业务信息"
            f"eg:检索时带上详细的问题内容"
        ),
    )

    query_tool_topo = QueryEngineTool.from_defaults(
        query_engine=topo_engine,
        name="topo",
        description=(
            f"关于电网拓扑中心的业务中台，包括了一、电网拓扑中心概述、二、拓扑基础分析服务群、三、拓扑高级分析服务 等业务信息"
            f"eg:检索时带上详细的问题内容"
        ),
    )
    query_tool_geo = QueryEngineTool.from_defaults(
        query_engine=geo_engine,
        name="geo",
        description=(
            f"关于电网图形中心的业务中台，包括了一、空间查询服务、二、专题图成图服务、三、Gis出图服务、专题图出图服务 等业务信息"
            f"eg:检索时带上详细的问题内容"
        ),
    )
    query_tool_asset = QueryEngineTool.from_defaults(
        query_engine=asset_engine,
        name="asset",
        description=(
            f"关于电网资产中心的业务中台，包括了资产信息查询服务 业务信息"
            f"eg:检索时带上详细的问题内容"
        ),
    )

    query_tool_psr = QueryEngineTool.from_defaults(
        query_engine=psr_engine,
        name="psr",
        description=(
            f"关于电网资源中心的业务中台，包括了资源查询服务 业务信息"
            f"eg:检索时带上详细的问题内容"
        ),
    )
    query_tool_base = QueryEngineTool.from_defaults(
        query_engine=base_engine,
        name="base",
        description=(
            f"关于电基础服务中心的业务中台，包括了3.1电网变更服务、3.2统一认证服务、3.3通用查询服务、3.4结构树服务、3.5权限服务、3.6人员组织查询服务 业务信息"
            f"eg:检索时带上详细的问题内容"
        ),
    )
    query_engine_tools = [
        query_tool_meas,
        query_tool_topo,
        query_tool_geo,
        query_tool_asset,
        query_tool_psr,
        query_tool_base,
    ]
    return query_engine_tools
