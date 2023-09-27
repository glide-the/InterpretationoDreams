from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.schema import HumanMessage
from langchain.schema.language_model import BaseLanguageModel

from dreamsboard.chains.prompts import (
    EDREAMS_PERSONALITY_TEMPLATE,
    STORY_BOARD_SCENE_TEMPLATE,
    STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE,
    EDREAMS_EVOLUTIONARY_TEMPLATE,
    DREAMS_GEN_TEMPLATE,
)
from dreamsboard.document_loaders import StructuredStoryboardCSVBuilder, KorLoader
from dreamsboard.generate.code_executor import CodeExecutor
from dreamsboard.generate.code_generate import BaseProgramGenerator, EngineProgramGenerator, AIProgramGenerator, \
    QueryProgramGenerator
from dreamsboard.generate.run_generate import CodeGeneratorBuilder


class StoryBoardDreamsGenerationChain(ABC):
    builder: StructuredStoryboardCSVBuilder
    dreams_guidance_chain: SequentialChain
    dreams_personality_chain: SequentialChain

    def __init__(self, csv_file_path: str,
                 dreams_guidance_chain: SequentialChain,
                 dreams_personality_chain: SequentialChain):
        self.builder = StructuredStoryboardCSVBuilder.form_builder(csv_file_path=csv_file_path)
        self.dreams_guidance_chain = dreams_guidance_chain
        self.dreams_personality_chain = dreams_personality_chain

    @classmethod
    def from_dreams_personality_chain(
            cls,
            llm: BaseLanguageModel,
            csv_file_path: str
    ) -> StoryBoardDreamsGenerationChain:
        # 03- 故事情境生成.txt STORY_BOARD_SCENE_TEMPLATE_Chain
        prompt_template1 = PromptTemplate(input_variables=["scene_content"],
                                          template=STORY_BOARD_SCENE_TEMPLATE)

        review_chain1 = LLMChain(llm=llm, prompt=prompt_template1, output_key="story_scenario_context")

        # 03-故事场景生成.txt
        prompt_template2 = PromptTemplate(input_variables=["story_board_summary_context"],
                                          template=STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE)
        review_chain2 = LLMChain(llm=llm, prompt=prompt_template2, output_key="scene_monologue_context")
        # 04-情感情景引导.txt
        prompt_template = PromptTemplate(input_variables=["story_board_summary_context",
                                                          "story_scenario_context",
                                                          "scene_monologue_context"],
                                         template=DREAMS_GEN_TEMPLATE)
        social_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="dreams_guidance_context")

        dreams_guidance_chain = SequentialChain(
            chains=[review_chain1, review_chain2, social_chain],
            input_variables=["scene_content", "story_board_summary_context"],
            # Here we return multiple variables
            output_variables=["dreams_guidance_context"],
            verbose=True)

        # 05-剧情总结.txt
        prompt_template05 = PromptTemplate(input_variables=["story_board_summary_context"],
                                           template=EDREAMS_EVOLUTIONARY_TEMPLATE)
        evolutionary_chain = LLMChain(llm=llm, prompt=prompt_template05, output_key="evolutionary_step")
        # 05-性格分析.txt
        prompt_template05 = PromptTemplate(input_variables=["evolutionary_step"],
                                           template=EDREAMS_PERSONALITY_TEMPLATE)
        personality_chain = LLMChain(llm=llm, prompt=prompt_template05, output_key="dreams_personality_context")

        dreams_personality_chain = SequentialChain(
            chains=[evolutionary_chain, personality_chain],
            input_variables=["story_board_summary_context"],
            # Here we return multiple variables
            output_variables=["dreams_personality_context"],
            verbose=True)
        return cls(csv_file_path=csv_file_path,
                   dreams_guidance_chain=dreams_guidance_chain,
                   dreams_personality_chain=dreams_personality_chain)

    def run(self) -> Dict[str, Any]:
        # 对传入的剧本台词转换成 scene_content
        self.builder.load()
        selected_columns = ["story_board_text", "story_board"]
        scene_content = self.builder.build_text(selected_columns)
        story_board_summary_context = self.builder.build_msg()

        dreams_guidance_personality_chain = SequentialChain(
            chains=[self.dreams_guidance_chain, self.dreams_personality_chain],
            input_variables=["scene_content", "story_board_summary_context"],
            # Here we return multiple variables
            output_variables=["dreams_guidance_context", "dreams_personality_context"],
            verbose=True)

        dreams_out = dreams_guidance_personality_chain({"scene_content": scene_content,
                                                        "story_board_summary_context": story_board_summary_context})

        return {"dreams_guidance_context": dreams_out["dreams_guidance_context"],
                "dreams_personality_context": dreams_out["dreams_personality_context"]}


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
        code_gen_builder = CodeGeneratorBuilder()

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
            print(guidance_question.step_description)
            _dreams_render_data = {
                'dreams_cosplay_role': '心理咨询工作者',
                'dreams_message': guidance_question.step_advice,
            }
            code_gen_builder.add_generator(QueryProgramGenerator.from_config(cfg={
                "dreams_query_code_file": "dreams_query_template.py-tpl",
                "render_data": _dreams_render_data,
            }))
            code_gen_builder.add_generator(EngineProgramGenerator.from_config(cfg={
                "engine_code_file": "engine_template.py-tpl",
            }))
            executor = code_gen_builder.build_executor()
            executor.execute()
            _ai_message = executor.chat_run()
            code_gen_builder.remove_last_generator()

            _ai_render_data = {
                'ai_message_content': _ai_message.content
            }
            code_gen_builder.add_generator(AIProgramGenerator.from_config(cfg={
                "ai_code_file": "ai_template.py-tpl",
                "render_data": _ai_render_data,
            }))

        return code_gen_builder

