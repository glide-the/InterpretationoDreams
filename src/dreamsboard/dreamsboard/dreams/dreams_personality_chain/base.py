from __future__ import annotations

import logging
import os
from abc import ABC
from typing import Any, Dict

from langchain.chains import SequentialChain
from langchain.chains.base import Chain
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

from dreamsboard.document_loaders import StructuredStoryboardCSVBuilder
from dreamsboard.dreams.dreams_personality_chain.prompts import (
    DREAMS_GEN_TEMPLATE,
    EDREAMS_EVOLUTIONARY_TEMPLATE,
    EDREAMS_PERSONALITY_TEMPLATE,
    STORY_BOARD_SCENE_TEMPLATE,
    STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


class StoryBoardDreamsGenerationChain(ABC):
    builder: StructuredStoryboardCSVBuilder
    dreams_guidance_chain: Chain
    dreams_personality_chain: Chain

    def __init__(
        self,
        csv_file_path: str,
        user_id: str,
        dreams_guidance_chain: Chain,
        dreams_personality_chain: Chain,
    ):
        self.builder = StructuredStoryboardCSVBuilder.form_builder(
            csv_file_path=csv_file_path
        )
        self.user_id = user_id
        self.dreams_guidance_chain = dreams_guidance_chain
        self.dreams_personality_chain = dreams_personality_chain

    @classmethod
    def from_dreams_personality_chain(
        cls,
        llm_runable: Runnable[LanguageModelInput, BaseMessage],
        csv_file_path: str,
        user_id: str = None,
    ) -> StoryBoardDreamsGenerationChain:
        # 03- 故事情境生成 `story_scenario_context`.txt STORY_BOARD_SCENE_TEMPLATE_Chain
        prompt_template1 = PromptTemplate(
            input_variables=["scene_content"], template=STORY_BOARD_SCENE_TEMPLATE
        )

        review_chain1 = prompt_template1 | llm_runable | StrOutputParser()

        # 03-故事场景生成 `scene_monologue_context`.txt
        prompt_template2 = PromptTemplate(
            input_variables=["story_board_summary_context", "user_id"],
            template=os.environ.get(
                "STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE",
                STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE,
            ),
        )
        review_chain2 = prompt_template2 | llm_runable | StrOutputParser()
        # 04-情感情景引导.txt
        prompt_template = PromptTemplate(
            input_variables=[
                "story_board_summary_context",
                "story_scenario_context",
                "scene_monologue_context",
                "user_id",
            ],
            template=os.environ.get("DREAMS_GEN_TEMPLATE", DREAMS_GEN_TEMPLATE),
        )
        social_chain = prompt_template | llm_runable | StrOutputParser()

        def wrapper_guidance_unit(_dict):
            return {
                # 中间变量包装至下一个pipline
                "story_scenario_context": _dict["story_scenario_context"],
                "scene_monologue_context": _dict["scene_monologue_context"],
                "user_id": _dict["user_id"],
                "scene_content": _dict["scene_content"],
                "story_board_summary_context": _dict["story_board_summary_context"],
            }

        def wrapper_guidance_output(_dict):
            return {
                # 中间变量全部打包输出
                "dreams_guidance_context": _dict["dreams_guidance_context"],
                "story_scenario_context": _dict["story_scenario_context"],
                "scene_monologue_context": _dict["scene_monologue_context"],
                "user_id": _dict["user_id"],
                "scene_content": _dict["scene_content"],
                "story_board_summary_context": _dict["story_board_summary_context"],
            }

        dreams_guidance_chain = (
            {
                "story_scenario_context": review_chain1,
                "scene_monologue_context": review_chain2,
                "user_id": lambda x: x["user_id"],
                "scene_content": lambda x: x["scene_content"],
                "story_board_summary_context": lambda x: x[
                    "story_board_summary_context"
                ],
            }
            | RunnableLambda(wrapper_guidance_unit)
            | {
                "dreams_guidance_context": social_chain,
                "story_scenario_context": lambda x: x["story_scenario_context"],
                "scene_monologue_context": lambda x: x["scene_monologue_context"],
                "user_id": lambda x: x["user_id"],
                "scene_content": lambda x: x["scene_content"],
                "story_board_summary_context": lambda x: x[
                    "story_board_summary_context"
                ],
            }
            | RunnableLambda(wrapper_guidance_output)
        )

        # 05-剧情总结 `evolutionary_step`.txt
        prompt_template04 = PromptTemplate(
            input_variables=["story_board_summary_context"],
            template=os.environ.get(
                "EDREAMS_EVOLUTIONARY_TEMPLATE", EDREAMS_EVOLUTIONARY_TEMPLATE
            ),
        )
        evolutionary_chain = prompt_template04 | llm_runable | StrOutputParser()
        # 05-性格分析 `dreams_personality_context`.txt
        prompt_template05 = PromptTemplate(
            input_variables=["evolutionary_step"],
            template=os.environ.get(
                "EDREAMS_PERSONALITY_TEMPLATE", EDREAMS_PERSONALITY_TEMPLATE
            ),
        )
        personality_chain = prompt_template05 | llm_runable | StrOutputParser()

        def wrapper_personality_unit(_dict):
            return {
                # 中间变量包装至下一个pipline
                "evolutionary_step": _dict["evolutionary_step"],
                "story_board_summary_context": _dict["story_board_summary_context"],
            }

        def wrapper_personality_output(_dict):
            return {
                # 中间变量全部打包输出
                "dreams_personality_context": _dict["dreams_personality_context"],
                "evolutionary_step": _dict["evolutionary_step"],
                "story_board_summary_context": _dict["story_board_summary_context"],
            }

        dreams_personality_chain = (
            {
                "story_board_summary_context": lambda x: x[
                    "story_board_summary_context"
                ],
                "evolutionary_step": evolutionary_chain,
            }
            | RunnableLambda(wrapper_personality_unit)
            | {
                "dreams_personality_context": personality_chain,
                "evolutionary_step": lambda x: x["evolutionary_step"],
                "story_board_summary_context": lambda x: x[
                    "story_board_summary_context"
                ],
            }
            | RunnableLambda(wrapper_personality_output)
        )

        return cls(
            csv_file_path=csv_file_path,
            user_id=user_id if user_id else "此来访者",
            dreams_guidance_chain=dreams_guidance_chain,
            dreams_personality_chain=dreams_personality_chain,
        )

    def run(self) -> Dict[str, Any]:
        # 对传入的剧本台词转换成 scene_content
        self.builder.load()
        selected_columns = ["story_board_role", "story_board_text", "story_board"]
        scene_content = self.builder.build_text(self.user_id, selected_columns)
        story_board_summary_context = self.builder.build_msg()

        dreams_guidance_personality_map_chain = RunnableParallel(
            dreams_guidance_context=self.dreams_guidance_chain,
            dreams_personality_context=self.dreams_personality_chain,
        )

        return dreams_guidance_personality_map_chain.invoke(
            {
                "scene_content": scene_content,
                "story_board_summary_context": story_board_summary_context,
                "user_id": self.user_id,
            }
        )

    def invoke_dreams_guidance_chain(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.dreams_guidance_chain.invoke(input)

    def invoke_dreams_personality_chain(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.dreams_personality_chain.invoke(input)
