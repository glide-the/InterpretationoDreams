from __future__ import annotations
from typing import List
from dreamsboard.engine.engine_builder import CodeGeneratorBuilder
from dreamsboard.engine.generate.code_generate import (
    BaseProgramGenerator,
    QueryProgramGenerator,
    AIProgramGenerator,
    EngineProgramGenerator,
)
from dreamsboard.document_loaders import StructuredStoryboardCSVBuilder, KorLoader
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel
import logging

from dreamsboard.engine.loading import load_store_from_storage
from dreamsboard.engine.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


class DreamsStepInfo:
    def __init__(self, step_advice, step_description):
        self.step_advice = step_advice
        self.step_description = step_description


class StructuredDreamsStoryboard:
    """
    对剧本和分析结果进行结构化，将开放问题与性格分析结果进行结合。生成情景扮演代码
    此过程如下
        对开放问题结果进行抽取，得到问题内容
        对性格分析结果进行抽取，得到性格分析结果
        根据剧本与任务性格基础情景扮演代码，在传入得到的问题，生成问题的答案
        增加系统提示词
        导出代码
    """

    def __init__(self,
                 builder: StructuredStoryboardCSVBuilder,
                 dreams_guidance_context: str,
                 dreams_personality_context: str,
                 kor_dreams_guidance_chain: LLMChain,
                 kor_dreams_personality_chain: LLMChain,
                 ):
        """

        :param builder: 剧本
        :param dreams_guidance_context: 开放问题
        :param dreams_personality_context: 性格分析结果
        """
        self.builder = builder
        self.dreams_guidance_context = dreams_guidance_context
        self.dreams_personality_context = dreams_personality_context
        self.kor_dreams_guidance_chain = kor_dreams_guidance_chain
        self.kor_dreams_personality_chain = kor_dreams_personality_chain
        self.storage_context = None

    @classmethod
    def form_builder(cls,
                     llm: BaseLanguageModel,
                     builder: StructuredStoryboardCSVBuilder,
                     dreams_guidance_context: str,
                     dreams_personality_context: str) -> StructuredDreamsStoryboard:
        kor_dreams_guidance_chain = KorLoader.form_kor_dreams_guidance_builder(llm=llm)
        kor_dreams_personality_chain = KorLoader.form_kor_dreams_personality_builder(llm=llm)

        return cls(builder=builder,
                   dreams_guidance_context=dreams_guidance_context,
                   dreams_personality_context=dreams_personality_context,
                   kor_dreams_guidance_chain=kor_dreams_guidance_chain,
                   kor_dreams_personality_chain=kor_dreams_personality_chain)

    def kor_dreams_guidance_context(self) -> List[DreamsStepInfo]:
        """
        对开放问题结果进行抽取，得到问题内容
        :return:
        """
        response = self.kor_dreams_guidance_chain.run(self.dreams_guidance_context)
        dreams_step_list = []
        if response.get('data') is not None and response.get('data').get('script') is not None:
            step_list = response.get('data').get('script')
            for step in step_list:
                dreams_step = DreamsStepInfo(step_advice=step.get('step_advice'),
                                             step_description=step.get('step_description'))
                dreams_step_list.append(dreams_step)

        return dreams_step_list

    def kor_dreams_personality_context(self) -> str:
        """
        对性格分析结果进行抽取，得到性格分析结果
        :return:
        """
        response = self.kor_dreams_personality_chain.run(self.dreams_personality_context)

        personality = ""
        if response.get('data') is not None and response.get('data').get('script') is not None:
            personality_list = response.get('data').get('script')
            # [{'personality': '具有情感表达和期待、注重个体快感、善于运用语义信息、对社会行为产生兴趣'}]},
            # 拼接personality 成一个字符串
            for item in personality_list:
                personality += item.get('personality') + "、"

        return personality

    def builder_base_cosplay_code(self) -> List[str]:
        """
        根据剧本与任务性格基础情景扮演代码，
        :return:
        """

        self.builder.load()
        # 创建一个字典，用于按照story_board组织内容和角色
        storyboard_dict = self.builder.build_dict()

        # 根据story_board组织内容和角色
        messages = []
        for storyboard, storyboard_data in storyboard_dict.items():
            # 格式化内容
            text = '，'.join(storyboard_data['story_board_text'])
            text += "。"
            messages.append(f"{storyboard_data['story_board_role'][0]}:「{text}」")

        return messages

    def loader_cosplay_builder(self) -> CodeGeneratorBuilder:
        code_gen_builder = CodeGeneratorBuilder.from_template(nodes=[])

        # 创建一个字典，用于按照story_board组织内容和角色
        storyboard_dict = self.builder.build_dict()
        # 获取第一个story_board_role属性的值
        cosplay_role = list(storyboard_dict.values())[0]['story_board_role'][0]

        guidance_questions = self.kor_dreams_guidance_context()
        personality = self.kor_dreams_personality_context()

        base_cosplay_message = self.builder_base_cosplay_code()
        _base_render_data = {
            'cosplay_role': cosplay_role,
            'personality': personality,
            'messages': base_cosplay_message
        }
        code_gen_builder.add_generator(BaseProgramGenerator.from_config(cfg={
            "code_file": "base_template.py-tpl",
            "render_data": _base_render_data,
        }))

        for guidance_question in guidance_questions:
            logger.info(f'{guidance_question.step_description}:{guidance_question.step_advice}')
            _dreams_render_data = {
                'dreams_cosplay_role': '心理咨询工作者',
                'dreams_message': guidance_question.step_advice,
            }
            code_gen_builder.add_generator(QueryProgramGenerator.from_config(cfg={
                "query_code_file": "dreams_query_template.py-tpl",
                "render_data": _dreams_render_data,
            }))
            code_gen_builder.add_generator(EngineProgramGenerator.from_config(cfg={
                "engine_code_file": "engine_template.py-tpl",
            }))
            executor = code_gen_builder.build_executor()
            executor.execute()
            _ai_message = executor.chat_run()

            logger.info(f'{guidance_question.step_description}:{_ai_message}')
            code_gen_builder.remove_last_generator()

            _ai_render_data = {
                'ai_message_content': _ai_message.content
            }
            code_gen_builder.add_generator(AIProgramGenerator.from_config(cfg={
                "ai_code_file": "ai_template.py-tpl",
                "render_data": _ai_render_data,
            }))

        return code_gen_builder
