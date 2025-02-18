import hashlib
import json
import logging
import os
import tempfile
from typing import List

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)
CACHE_DIR = os.path.join(tempfile.gettempdir(), "query_cache")  # 自定义缓存目录
# 定义接口 URL
SEARCH_PAPERS_URL = "http://180.184.65.98:38880/atomgit/search_papers"
QUERY_BY_PAPER_ID_URL = "http://180.184.65.98:38880/atomgit/query_by_paper_id"


def search_papers(query, top_k=5):
    """
    根据查询文本 search_papers 接口进行模糊查询，返回论文 ID 列表
    """
    params = {"query": query, "top_k": top_k}
    response = requests.get(SEARCH_PAPERS_URL, params=params)

    if response.status_code == 200:
        # 返回的 JSON 数据，假设是一个包含论文信息的数组
        return response.json()  # 返回论文的列表
    else:
        logger.info(f"Error: {response.status_code}")
        return None


def query_by_paper_id(paper_id, top_k=5):
    """
    根据 paper_id 调用 query_by_paper_id 接口，获取该论文的详细信息
    """
    params = {"paper_id": paper_id, "top_k": top_k}
    response = requests.get(QUERY_BY_PAPER_ID_URL, params=params)

    if response.status_code == 200:
        # 返回的 JSON 数据，假设是论文的详细信息
        return response.json()  # 返回论文的详细信息
    else:
        logger.info(f"Error: {response.status_code}")
        return None


def ensure_cache_dir():
    """确保缓存目录存在"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def get_query_hash(query):
    """根据 query 生成唯一的哈希值"""
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def get_cache_file_path(query):
    """获取 query 对应的缓存文件路径"""
    query_hash = get_query_hash(query)
    return os.path.join(CACHE_DIR, f"{query_hash}.json")


def check_cache(query):
    """检查是否存在 query 的缓存文件"""
    file_path = get_cache_file_path(query)
    if os.path.exists(file_path):
        logger.info(f"Cache hit file_path: {file_path} for query: {query}")
        with open(file_path, "r", encoding="utf-8") as cache_file:
            return json.load(cache_file)  # 返回缓存内容
    return None


def save_to_cache(query, data):
    """将数据保存到 query 的缓存文件中"""
    ensure_cache_dir()
    file_path = get_cache_file_path(query)
    with open(file_path, "w", encoding="utf-8") as cache_file:
        json.dump(data, cache_file, ensure_ascii=False, indent=4)
        logger.info(f"Data cached for query: {query} at {file_path}")


def prepare_properties(paper_details: List[dict]):
    """
    将 paper_details 数据转化为向量数据库插入所需的 properties 格式
    """
    properties_list = []

    for paper in paper_details:
        properties = {
            "ref_id": paper.get("id"),  # 可能是另一个 ID，具体根据实际情况
            "paper_id": paper.get("paper_id"),
            "paper_title": paper.get("paper_title"),
            "chunk_id": paper.get("chunk_id"),
            "chunk_text": paper.get("chunk_text"),
            "original_filename": paper.get(
                "original_filename", ""
            ),  # 默认空字符串，如果没有提供
        }
        properties_list.append(properties)

    return properties_list


def exe_query(query, top_k):
    cached_data = check_cache(query)
    if cached_data:
        logger.info(f"Using cached data for query: {query}")
        return cached_data  # 如果有缓存则直接返回
    # 第一步：模糊查询论文
    logger.info(f"Searching papers for query: {query}")
    papers = search_papers(query, top_k)

    if papers:
        logger.info(f"Found {len(papers)} papers.")

        # 获取所有的 paper_id，并去重
        paper_ids = set()  # 使用 set 去重
        for paper in papers:
            paper_id = paper.get("entity", {}).get("paper_id")
            if paper_id:
                paper_ids.add(paper_id)  # 添加到 set 中，自动去重

        # 输出去重后的 paper_id 数量
        logger.info(f"Found {len(paper_ids)} unique paper IDs.")

        # 查询每个 unique paper_id 的详细信息
        all_paper_details = []
        for paper_id in paper_ids:
            logger.info(f"Fetching details for paper ID: {paper_id}")
            paper_details = query_by_paper_id(paper_id, top_k)
            if paper_details:
                all_paper_details.extend(paper_details)  # 假设返回的是一个列表
            else:
                logger.info(f"Failed to fetch details for paper ID: {paper_id}")

        # 将获取的 paper_details 转换为向量数据库可插入的格式
        properties_list = prepare_properties(all_paper_details)
        # 保存 properties_list 到临时文件
        save_to_cache(query, properties_list)
        return properties_list
    else:
        logger.info("No papers found.")
        return []
