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
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from pandas import DataFrame


@dataclass
class QuestionContext:
    def __init__(self, ref_id: str, chunk_id: str, score: float, text: str):
        self.ref_id = ref_id
        self.chunk_id = chunk_id
        self.score = score
        self.text = text

    def to_dict(self):
        return {
            "ref_id": self.ref_id,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
        }


# 定义链表节点类
class LinkedListNode:
    def __init__(
        self,
        task_step_id: str,
        shot_number: int,
        scene_number: str,
        start_task_context: str = None,
        aemo_representation_context: str = None,
        task_step_name: str = None,
        task_step_description: str = None,
        task_step_level: str = None,
        task_step_question: str = None,
        task_step_question_context: List[QuestionContext] = None,
        task_step_question_answer: str = None,
        ref_task_step_id: str = None,
    ):
        self.task_step_id = task_step_id
        self.shot_number = shot_number
        self.scene_number = scene_number
        self.start_task_context = start_task_context
        self.aemo_representation_context = aemo_representation_context
        self.task_step_name = task_step_name
        self.task_step_description = task_step_description
        self.task_step_level = task_step_level
        self.task_step_question = task_step_question
        self.task_step_question_context = task_step_question_context
        self.task_step_question_answer = task_step_question_answer
        self.ref_task_step_id = ref_task_step_id
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
            shot_number = self.prev_node.shot_number + 1

        for index, data in enumerate(json_data):
            scene_number = "story_board" + str(index)
            task_step_id = data["id_"]
            start_task_context = data["start_task_context"]
            aemo_representation_context = data["aemo_representation_context"]
            task_step_name = data["task_step_name"]
            task_step_description = data["task_step_description"]
            task_step_level = data["task_step_level"]
            task_step_question = data["task_step_question"]
            task_step_question_context = [
                QuestionContext(
                    ref_id=context.get("ref_id", ""),
                    chunk_id=context.get("chunk_id", ""),
                    score=context.get("score", ""),
                    text=context.get("text", ""),
                )
                for context in data["task_step_question_context"]
            ]
            task_step_question_answer = data["task_step_question_answer"]
            ref_task_step_id = data["ref_task_step_id"]

            node = LinkedListNode(
                task_step_id,
                shot_number,
                scene_number,
                start_task_context,
                aemo_representation_context,
                task_step_name,
                task_step_description,
                task_step_level,
                task_step_question,
                task_step_question_context,
                task_step_question_answer,
                ref_task_step_id,
            )
            if self.prev_node:
                self.prev_node.next = node
                node.prev = self.prev_node

            else:
                self.head = node

            node.head = self.head
            self.prev_node = node
            shot_number += 1

    def get_task_step_node(self, task_step_id: str) -> LinkedListNode:
        """
        获取任务步骤节点
        """
        current_parse_node = self.head
        while current_parse_node is not None:
            if current_parse_node.task_step_id == task_step_id:
                return current_parse_node
            current_parse_node = current_parse_node.next
        return None

    def parse_table(self) -> DataFrame:
        """
        输出格式为
            任务步骤编号	分镜编号	场景编号    开始任务	任务总体描述	任务步骤名称	任务步骤描述	任务步骤层级	任务步骤问题	任务步骤问题上下文	任务步骤问题答案	参考任务步骤编号
        :return: DataFrame
        """
        table_data = []
        current_parse_node = self.head
        while current_parse_node is not None:
            task_context = [
                context.to_dict()
                for context in current_parse_node.task_step_question_context
            ]  # 遍历并转换每个对象为字典

            row = [
                current_parse_node.task_step_id,
                current_parse_node.shot_number,
                current_parse_node.scene_number,
                current_parse_node.start_task_context,
                current_parse_node.aemo_representation_context,
                current_parse_node.task_step_name,
                current_parse_node.task_step_description,
                current_parse_node.task_step_level,
                current_parse_node.task_step_question,
                task_context,
                current_parse_node.task_step_question_answer,
                current_parse_node.ref_task_step_id,
            ]
            table_data.append(row)
            current_parse_node = current_parse_node.next

        table = pd.DataFrame(
            table_data,
            columns=[
                "task_step_id",
                "shot_number",
                "scene_number",
                "start_task_context",
                "aemo_representation_context",
                "task_step_name",
                "task_step_description",
                "task_step_level",
                "task_step_question",
                "task_step_question_context",
                "task_step_question_answer",
                "ref_task_step_id",
            ],
        )
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
                f"task_step_id: {row.task_step_id},"
                f"scene_number: {row.scene_number},"
                f"shot_number: {row.shot_number}\n\n"
                f"start_task_context: {row.start_task_context}\n\n"
                f"aemo_representation_context: {row.aemo_representation_context}\n\n"
                f"task_step_name: {row.task_step_name}\n\n"
                f"task_step_description: {row.task_step_description}\n\n"
                f"task_step_level: {row.task_step_level}\n\n"
                f"task_step_question: {row.task_step_question}\n\n"
                f"task_step_question_context: {row.task_step_question_context}\n\n"
                f"task_step_question_answer: {row.task_step_question_answer}\n\n"
                f"ref_task_step_id: {row.ref_task_step_id}\n\n"
            )

            structured_storyboard_json = row.to_json(orient="records", indent=4)
            doc = Document(
                page_content=raw_transcript_with_meta_info,
                metadata=structured_storyboard_json,
            )

            documents.append(doc)

        return documents
