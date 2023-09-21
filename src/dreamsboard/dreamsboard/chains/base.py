from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.schema.language_model import BaseLanguageModel

from dreamsboard.chains.prompts import (
    EDREAMS_PERSONALITY_TEMPLATE,
    STORY_BOARD_SCENE_TEMPLATE,
    STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE,
    EDREAMS_EVOLUTIONARY_TEMPLATE,
    DREAMS_GEN_TEMPLATE,
)
from dreamsboard.document_loaders.csv_structured_storyboard_loader import StructuredStoryboardCSVBuilder


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
        prompt_template05 = PromptTemplate(input_variables=["scene_content"],
                                           template=EDREAMS_EVOLUTIONARY_TEMPLATE)
        evolutionary_chain = LLMChain(llm=llm, prompt=prompt_template05, output_key="evolutionary_step")
        # 05-性格分析.txt
        prompt_template05 = PromptTemplate(input_variables=["evolutionary_step"],
                                           template=EDREAMS_PERSONALITY_TEMPLATE)
        personality_chain = LLMChain(llm=llm, prompt=prompt_template05, output_key="dreams_personality_context")

        dreams_personality_chain = SequentialChain(
                chains=[evolutionary_chain, personality_chain],
                input_variables=["scene_content"],
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

        selected_story_board_columns = ["story_board_role", "story_board_text", "story_board"]
        story_board_summary_context = self.builder.build_text(selected_story_board_columns)

        dreams_guidance_out = self.dreams_guidance_chain({"scene_content": scene_content,
                                                          "story_board_summary_context": story_board_summary_context})
        dreams_personality_out = self.dreams_personality_chain({"scene_content": scene_content})
        return {"dreams_guidance_context": dreams_guidance_out["dreams_guidance_context"],
                "dreams_personality_context": dreams_personality_out["dreams_personality_context"]}

