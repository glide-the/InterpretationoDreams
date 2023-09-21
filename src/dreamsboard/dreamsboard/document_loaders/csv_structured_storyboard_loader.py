"""
加载csv文件
输入
空白	内容	开始时间	结束时间	分镜
七七	今天是温柔长裙风	0:00:00,560	0:00:02,720	0
七七	宝宝,你再不来我家找我玩的话	0:00:02,720	0:00:06,340	1


输出格式为
[{
start_point_len:0:00:00,560,
end_point_len:0:00:02,720,
story_board:"0",
story_board_text:"今天是温柔长裙风",
story_board_role:"七七",
},
{
start_point_len:0:00:00,560,
end_point_len:0:00:02,720,
story_board:"1",
story_board_text:"宝宝,你再不来我家找我玩的话",
story_board_role:"七七",
}]


"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
import csv


class Metadata:
    def __init__(self, name, label):
        self.name = name  # 属性名称
        self.label = label  # 属性标签


class StructuredStoryboard:
    def __init__(self, start_point_len, end_point_len, story_board, story_board_text, story_board_role):
        self.start_point_len = (start_point_len, "开始时间")
        self.end_point_len = (end_point_len, "结束时间")
        self.story_board = (story_board, "分镜")
        self.story_board_text = (story_board_text, "内容")
        self.story_board_role = (story_board_role, "角色")


class StructuredStoryboardCSVBuilder(ABC):
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.data = []

    @classmethod
    def form_builder(cls, csv_file_path: str) -> StructuredStoryboardCSVBuilder:

        return cls(csv_file_path=csv_file_path)

    def load(self):
        # 清空现有数据
        self.data = []

        # 打开CSV文件并读取数据
        with open(self.csv_file_path, newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file, delimiter=',', quotechar='"')
            next(csv_reader)  # 跳过标题行

            for row in csv_reader:
                if len(row) == 5:  # 确保每行有5个字段
                    role, text, start_time, end_time, storyboard = row
                    start_point_len = start_time
                    end_point_len = end_time

                    # 创建StructuredStoryboard对象
                    structured_storyboard = StructuredStoryboard(start_point_len, end_point_len, storyboard, text, role)

                    # 添加到数据列表
                    self.data.append(structured_storyboard)

    def build_text(self, columns_to_select):
        # 根据传入的列名列表筛选数据
        selected_data = []
        for item in self.data:
            selected_item = {}
            for column in columns_to_select:
                if hasattr(item, column):
                    selected_item[column] = getattr(item, column)[0]
            selected_data.append(selected_item)

        # 获取列名对应的标签
        metadatas = []
        for column in columns_to_select:
            if hasattr(self.data[0], column):
                metadata = getattr(self.data[0], column)[1]
                metadatas.append(metadata)

        # 输出格式化的文本
        formatted_text = "    ".join(metadatas)
        formatted_text += "\n"
        for item in selected_data:
            # 遍历每个键值对并输出
            formatted_text += '\t'.join([item[field_name] for field_name in columns_to_select])
            if item != selected_data[-1]:
                formatted_text += "\n"

        return formatted_text


