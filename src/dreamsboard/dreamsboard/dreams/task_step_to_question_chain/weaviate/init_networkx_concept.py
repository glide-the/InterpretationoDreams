# Import necessary libraries
import copy
import json
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
from IPython.core.display import HTML, display
from matplotlib import rcParams
from pyvis.network import Network

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)
# 设置 HTTP 和 HTTPS 代理
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


# 设置中文字体为 SimHei（黑体）
rcParams["font.sans-serif"] = ["DejaVu Serif"]  # Example

# rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# Function to create a graph from semantic path data
def create_caption_graph(data):
    G = nx.DiGraph()
    for i, node in enumerate(data):
        concept = node["concept"]
        G.add_node(
            concept,
            distanceToQuery=node["distanceToQuery"],
            distanceToResult=node["distanceToResult"],
        )
        if i < len(data) - 1:
            next_concept = data[i + 1]["concept"]
            G.add_edge(concept, next_concept, weight=node["distanceToNext"])
    return G


# 图形绘制函数，带上下翻译对比
def plot_graph_with_caption(G, title, caption, pos=None, translate=True):
    translated_title, translated_caption = title, caption

    if translate:
        # 自动翻译标题和说明
        translated_title = title
        # translated_caption = GoogleTranslator(source="en", target="zh-CN").translate(caption)
        translated_caption = caption

    plt.figure(figsize=(10, 10))
    pos = pos or nx.spring_layout(G, seed=42)  # 节点布局
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=2000,
        font_size=10,
    )
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={k: f"{v:.3f}" for k, v in labels.items()}
    )

    # 显示翻译和原文对比
    plt.title(f"{translated_title}\n(Original: {title})", fontsize=14, pad=20)
    plt.figtext(
        0.5,
        0.01,
        f"Translated Caption: {translated_caption}\n(Original Caption: {caption})",
        wrap=True,
        horizontalalignment="center",
        fontsize=12,
    )
    plt.show()


# Updated caption graph function
def create_and_plot_all(data_list):
    for i, context in enumerate(data_list):
        semantic_path = context["_additional"]["semanticPath"]["path"]
        caption = context["chunkText"]
        graph = create_caption_graph(semantic_path)
        title = f"Semantic Path Visualization {i+1}"
        plot_graph_with_caption(graph, title, caption)


# Function to find root nodes and their properties
def find_root_nodes(data):
    # Create a directed graph using NetworkX
    G = nx.DiGraph()
    node_properties = []

    for context in data:
        path = context["_additional"]["semanticPath"]["path"]
        for i, node in enumerate(path):
            concept = node["concept"]

            # Add node with its properties
            if concept not in G.nodes:
                G.add_node(concept)
            if i < len(path) - 1:
                next_concept = path[i + 1]["concept"]
                if next_concept not in G.nodes:
                    G.add_node(next_concept)
                G.add_edge(concept, next_concept)

    # Find nodes with in-degree 0 (root nodes)
    root_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]

    return root_nodes


# Function to find concepts with more than 3 outgoing edges
def find_high_outdegree_concepts(data, edge_threshold=3):
    # Create a directed graph using NetworkX
    G = nx.DiGraph()

    for context in data:
        path = context["_additional"]["semanticPath"]["path"]
        for i, node in enumerate(path):
            concept = node["concept"]
            if concept not in G.nodes:
                G.add_node(concept)
            if i < len(path) - 1:
                next_concept = path[i + 1]["concept"]
                if next_concept not in G.nodes:
                    G.add_node(next_concept)
                G.add_edge(concept, next_concept)

    # Find nodes with outgoing edges greater than the threshold
    high_outdegree_concepts = [
        node for node, out_degree in G.out_degree() if out_degree > edge_threshold
    ]
    return high_outdegree_concepts


# Function to find nodes with in-degree greater than a threshold
def find_high_indegree_nodes(data, threshold=3):
    # Create a directed graph using NetworkX
    G = nx.DiGraph()
    for context in data:
        path = context["_additional"]["semanticPath"]["path"]
        for i, node in enumerate(path):
            concept = node["concept"]

            # Add node with its properties
            if concept not in G.nodes:
                G.add_node(concept)
            if i < len(path) - 1:
                next_concept = path[i + 1]["concept"]
                if next_concept not in G.nodes:
                    G.add_node(next_concept)
                G.add_edge(concept, next_concept)

    # Find nodes with in-degree greater than the threshold
    high_indegree_nodes = [
        node for node, in_degree in G.in_degree() if in_degree > threshold
    ]
    return high_indegree_nodes


def get_color(
    concept, root_nodes_with_concept, high_outdegree_concepts, high_indegree_concepts
):
    # 高亮根节点
    if concept in root_nodes_with_concept:
        color = "green"  # 绿色表示根节点
    # 高亮出度较大的节点
    elif concept in high_outdegree_concepts:
        color = "red"  # 红色表示出度较大的节点
    elif concept in high_indegree_concepts:
        color = "blue"  # 蓝色表示出度较大的节点
    else:
        color = "lightblue"  # 默认节点颜色

    return color


# 修改 create_interactive_graph 函数以标注特定节点
def create_interactive_graph(
    data,
    root_nodes_with_concept,
    high_outdegree_concepts,
    high_indegree_concepts,
    filename="semantic_path_interactive.html",
):
    net = Network(height="750px", width="100%", directed=True)

    for context in data:
        # 直接使用原始路径数据
        path = context["_additional"]["semanticPath"]["path"]
        nodes_added = set()  # Track added nodes
        for i, node in enumerate(path):
            concept = node["concept"]

            if concept not in nodes_added:
                color = get_color(
                    concept,
                    root_nodes_with_concept,
                    high_outdegree_concepts,
                    high_indegree_concepts,
                )
                net.add_node(concept, label=concept, color=color)
                nodes_added.add(concept)

            if i < len(path) - 1:
                next_concept = path[i + 1]["concept"]
                # Ensure the next concept exists before adding the edge
                if next_concept not in nodes_added:
                    color = get_color(
                        next_concept,
                        root_nodes_with_concept,
                        high_outdegree_concepts,
                        high_indegree_concepts,
                    )
                    net.add_node(next_concept, label=next_concept, color=color)
                    nodes_added.add(next_concept)
                net.add_edge(
                    concept,
                    next_concept,
                    title=f"Distance: {node['distanceToNext']:.3f}",
                )

    # 保存 HTML 文件
    net.write_html(filename)

    # 在 Jupyter 中显示 HTML
    # iframe = f'<iframe src="{filename}" width="100%" height="750px" frameborder="0"></iframe>'
    # display(HTML(iframe))


# Function to find concepts with more than 3 outgoing edges
def create_G(
    data, root_nodes_with_concept, high_outdegree_concepts, high_indegree_concepts
):
    # 创建有向图
    G = nx.DiGraph()

    for context in data:
        path = context["_additional"]["semanticPath"]["path"]
        refId = context.get("refId", None)  # 从context中获取refId字段
        nodes_added = set()  # 用于在同一个context中避免重复添加同一节点属性

        for i, node in enumerate(path):
            concept = node["concept"]

            # 如果节点在全局图中不存在，则新建节点并添加属性
            if concept not in G:
                color = get_color(
                    concept,
                    root_nodes_with_concept,
                    high_outdegree_concepts,
                    high_indegree_concepts,
                )
                # 将refId信息存入节点属性中。如果同一节点在多个context中出现，可以存成列表/集合
                G.add_node(
                    concept, label=concept, color=color, refId=[refId] if refId else []
                )
            else:
                # 如果节点已存在，且有新的refId信息，合并进去（去重）
                if refId and refId not in G.nodes[concept]["refId"]:
                    G.nodes[concept]["refId"].append(refId)

            nodes_added.add(concept)

            if i < len(path) - 1:
                next_concept = path[i + 1]["concept"]
                if next_concept not in G:
                    color = get_color(
                        next_concept,
                        root_nodes_with_concept,
                        high_outdegree_concepts,
                        high_indegree_concepts,
                    )
                    G.add_node(
                        next_concept,
                        label=next_concept,
                        color=color,
                        refId=[refId] if refId else [],
                    )
                else:
                    # 如果下一个概念节点已存在，更新refId信息
                    if refId and refId not in G.nodes[next_concept]["refId"]:
                        G.nodes[next_concept]["refId"].append(refId)

                # 增加边的属性（这里已有Distance信息）
                G.add_edge(
                    concept,
                    next_concept,
                    title=f"Distance: {node['distanceToNext']:.3f}",
                )

    return G


def find_simple_path(G, root_nodes_with_concept, high_outdegree_concepts):
    # 筛选出那些包含红色节点的路径
    result_paths = []
    for root in root_nodes_with_concept:
        for out in high_outdegree_concepts:
            if nx.has_path(G, root, out):
                # 寻找从 "root" 到 "out" 的所有简单路径
                paths = nx.all_shortest_paths(G, source=root, target=out)

                for path in paths:
                    # 判断路径中是否存在 color='red' 的节点
                    if any(G.nodes[node].get("color") == "red" for node in path):
                        if len(path) == 3:
                            result_paths.append(path)
            else:
                logger.info(f"No path between {root} and {out}")
                continue

    return result_paths
