import logging

import langchain
from langchain_core.prompt_values import StringPromptValue
from langchain_core.runnables import RunnableLambda, RunnableParallel

from dreamsboard.dreams.coplay_analysis_md.prompts import COSPLAY_ANALYSIS_MD_TEMPLATE
from dreamsboard.dreams.dreams_personality_chain.prompts import DREAMS_GEN_TEMPLATE
from dreamsboard.engine.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_1 = langchain.PromptTemplate(
    input_variables=[
        "story_board_summary_context",
        "story_scenario_context",
        "scene_monologue_context",
        "user_id",
    ],
    template=DREAMS_GEN_TEMPLATE,
)
_PROMPT_TEMPLATE_2 = langchain.PromptTemplate(
    input_variables=[
        "cosplay_role",
        "metadata",
        "story_scenario_context",
        "scene_monologue_context",
        "dreams_gen_text",
        "dreams_guidance_context",
        "evolutionary_step",
        "dreams_personality_context",
    ],
    template=COSPLAY_ANALYSIS_MD_TEMPLATE,
)


class CosplayAnalysisMD:
    def __init__(
        self,
        cosplay_role: str,
        source_url: str,
        keyframe: str,
        keyframe_path: str,
        storage_keyframe: str,
        storage_keyframe_path: str,
    ):
        self.cosplay_role = cosplay_role
        self.source_url = source_url
        self.keyframe = keyframe
        self.keyframe_path = keyframe_path
        self.storage_keyframe = storage_keyframe
        self.storage_keyframe_path = storage_keyframe_path
        self.storage_context = StorageContext.from_defaults(
            persist_dir=storage_keyframe_path
        )

    def format_md(self) -> StringPromptValue:
        def wrapper_guidance_unit(dict_input: dict):
            # 获取analysis_all第一个
            first_key = list(
                self.storage_context.dreams_analysis_store.analysis_all.keys()
            )[0]
            analysis = self.storage_context.dreams_analysis_store.analysis_all[
                first_key
            ]

            return analysis.to_dict()

        def wrapper_dreams_gen_unit(dict_input: StringPromptValue):
            # 获取analysis_all第一个
            first_key = list(
                self.storage_context.dreams_analysis_store.analysis_all.keys()
            )[0]
            analysis = self.storage_context.dreams_analysis_store.analysis_all[
                first_key
            ]

            return {
                "cosplay_role": self.cosplay_role,
                "dreams_gen_text": dict_input.text,
                "source_url": self.source_url,
                "keyframe": self.keyframe,
                "keyframe_path": self.keyframe_path,
                "storage_keyframe": self.storage_keyframe,
                "storage_keyframe_path": self.storage_keyframe_path,
                **analysis.to_dict(),
            }

        chain = (
            RunnableLambda(wrapper_guidance_unit)
            | _PROMPT_TEMPLATE_1
            | RunnableLambda(wrapper_dreams_gen_unit)
            | _PROMPT_TEMPLATE_2
        )

        out = chain.invoke({})
        return out

    def write_md(self, output_path: str) -> StringPromptValue:
        md = self.format_md()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md.text)
        logger.info(f"Write MD to {output_path}")
        return md
