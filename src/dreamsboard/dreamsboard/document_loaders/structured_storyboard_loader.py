"""
处理分镜结构化信息根据分镜画面，解析分镜中存在的信息、
输入为json结构数据
[{
time_point_len:00:00:01,
perv_story_board:""
story_board:"/story_board0.png"
next_story_board:"/story_board1.png"
},
{
time_point_len:00:00:05,
perv_story_board:"/story_board0.png"
story_board:"/story_board1.png"
next_story_board:"/story_board2.png"
}]

输出格式为
分镜编号	场景编号	时间点	描述	备注
Shot 1	story_board0	00:00:01	描述分镜细节 备注
Shot 2	story_board2	00:00:05	描述分镜细节 备注


"""
from abc import ABC
from typing import List, Dict, Any
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
import pandas as pd
from pandas import DataFrame


# 定义链表节点类
class LinkedListNode:
    def __init__(self, 
                shot_number: int, 
                scene_number: str, 
                start_task_context: str=None, 
                aemo_representation_context: str=None,
                task_step_name: str=None, 
                task_step_description: str=None, 
                task_step_level: str=None
        ):
        self.shot_number = shot_number
        self.scene_number = scene_number
        self.start_task_context = start_task_context
        self.aemo_representation_context = aemo_representation_context
        self.task_step_name = task_step_name
        self.task_step_description = task_step_description
        self.task_step_level = task_step_level
        self.head: LinkedListNode = None
        self.prev: LinkedListNode = None
        self.next: LinkedListNode = None


class StructuredStoryboard:
    """
    解析json数组中整理好的镜头，实现链表结构的处理操作
    """
    head: LinkedListNode = None
    prev_node: LinkedListNode = None

    def __init__(self, json_data: List[Dict[str, Any]]):
        """Initialize with  json_data."""
        self._parse_dialogue(json_data)

    def _parse_dialogue(self, json_data: List[Dict[str, Any]]) -> None:
        """
        解析JSON数组并创建链表
        """
        if self.prev_node is None:
            shot_number = 1
        else:
            shot_number = self.prev_node.shot_number+1

        for index, data in enumerate(json_data):
            scene_number = "story_board" + str(index)
            start_task_context = data["start_task_context"]
            aemo_representation_context = data["aemo_representation_context"]
            task_step_name = data["task_step_name"]
            task_step_description = data["task_step_description"]
            task_step_level = data["task_step_level"]

            node = LinkedListNode(shot_number, scene_number, start_task_context, aemo_representation_context, task_step_name, task_step_description, task_step_level)
            if self.prev_node:
                self.prev_node.next = node
                node.prev = self.prev_node

            else:
                self.head = node

            node.head = self.head
            self.prev_node = node
            shot_number += 1

    def parse_table(self) -> DataFrame:
        """
        输出格式为
            分镜编号	场景编号    开始任务	任务总体描述	任务步骤名称	任务步骤描述	任务步骤层级
        :return: DataFrame
        """
        table_data = []
        current_parse_node = self.head
        while current_parse_node is not None:
            row = [
                current_parse_node.shot_number,
                current_parse_node.scene_number,
                current_parse_node.start_task_context,
                current_parse_node.aemo_representation_context,
                current_parse_node.task_step_name,
                current_parse_node.task_step_description,
                current_parse_node.task_step_level
            ]
            table_data.append(row)
            current_parse_node = current_parse_node.next

        table = pd.DataFrame(table_data, columns=["shot_number", "scene_number", "start_task_context", "aemo_representation_context", "task_step_name", "task_step_description", "task_step_level"])
        return table


class StructuredStoryboardLoader(BaseLoader, ABC):
    structured_storyboard: StructuredStoryboard

    def __init__(self, structured_storyboard: StructuredStoryboard):
        """Initialize with  json_data."""
        self.structured_storyboard = structured_storyboard

    def load(self) -> List[Document]:
        documents = []

        table = self.structured_storyboard.parse_table()
        for row in table.itertuples():

            raw_transcript_with_meta_info = (
                f"scene_number: {row.scene_number},"
                f"shot_number: {row.shot_number}\n\n"
                f"start_task_context: {row.start_task_context}\n\n"
                f"aemo_representation_context: {row.aemo_representation_context}\n\n"
                f"task_step_name: {row.task_step_name}\n\n"
                f"task_step_description: {row.task_step_description}\n\n"
                f"task_step_level: {row.task_step_level}\n\n"
            )

            structured_storyboard_json = row.to_json(orient='records',  indent=4)
            doc = Document(page_content=raw_transcript_with_meta_info, metadata=structured_storyboard_json)

            documents.append(doc)

        return documents

