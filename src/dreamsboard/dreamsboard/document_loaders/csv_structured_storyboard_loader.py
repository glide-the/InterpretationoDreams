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
from abc import ABC
import csv


class Metadata:
    def __init__(self, name, label):
        self.name = name  # 属性名称
        self.label = label  # 属性标签


class StructuredStoryboard:
    def __init__(self, start_point_len, end_point_len, story_board, story_board_text, story_board_role):
        self.start_point_len = Metadata(start_point_len, "开始时间")
        self.end_point_len = Metadata(end_point_len, "结束时间")
        self.story_board = Metadata(story_board, "分镜")
        self.story_board_text = Metadata(story_board_text, "内容")
        self.story_board_role = Metadata(story_board_role, "角色")


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

    def build_text(self, columns_to_select) -> str:
        # 根据传入的列名列表筛选数据
        selected_data = []
        for item in self.data:
            selected_item = {}
            for column in columns_to_select:
                if hasattr(item, column):
                    selected_item[column] = getattr(item, column).name
            selected_data.append(selected_item)

        # 获取列名对应的标签
        metadatas = []
        for column in columns_to_select:
            if hasattr(self.data[0], column):
                metadata = getattr(self.data[0], column).label
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

    def build_dict(self) -> dict:
        """
        把StructuredStoryboard的列表 转换为story_board字典
        (story_board:[])
        :return:
        """

        # 创建一个字典，用于按照story_board组织内容和角色
        storyboard_dict = {}

        # 遍历StructuredStoryboard对象并将其按照story_board分组
        for storyboard in self.data:
            if storyboard.story_board.name in storyboard_dict:
                # 如果已经存在该story_board的组，追加内容和角色
                storyboard_dict[storyboard.story_board.name]['story_board_text'].append(
                    storyboard.story_board_text.name)
                storyboard_dict[storyboard.story_board.name]['story_board_role'].append(
                    storyboard.story_board_role.name)
            else:
                # 如果还没有该story_board的组，创建一个新的组
                storyboard_dict[storyboard.story_board.name] = {
                    'story_board_text': [storyboard.story_board_text.name],
                    'story_board_role': [storyboard.story_board_role.name]
                }

        return storyboard_dict

    def build_msg(self) -> str:
        """
        把StructuredStoryboard的列表将story_board_text和story_board_role按照story_board顺序拼接为下面内容
        story_board_role:「相同story_board的story_board_text」
        :return:
        """

        # 创建一个字典，用于按照story_board组织内容和角色
        storyboard_dict = self.build_dict()

        # 根据story_board组织内容和角色
        formatted_text = ""
        for storyboard, storyboard_data in storyboard_dict.items():
            # 格式化内容
            text = '，'.join(storyboard_data['story_board_text'])
            text += "。"
            formatted_text += f"{storyboard_data['story_board_role'][0]}:「{text}」"
            # 判断是否是最后一个storyboard
            if storyboard != list(storyboard_dict.keys())[-1]:
                formatted_text += "\n"

        return formatted_text
