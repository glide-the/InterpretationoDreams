import logging

from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from dreamsboard.document_loaders import StructuredStoryboardCSVBuilder
from dreamsboard.dreams.builder_cosplay_code.base import StructuredDreamsStoryboard
from dreamsboard.dreams.dreams_personality_chain.base import StoryBoardDreamsGenerationChain
import langchain

from dreamsboard.engine.dreams_personality.dreams_personality import DreamsPersonalityNode
from dreamsboard.engine.generate.code_generate import QueryProgramGenerator, EngineProgramGenerator, AIProgramGenerator
from dreamsboard.engine.loading import load_store_from_storage
from dreamsboard.engine.storage.dreams_analysis_store.simple_dreams_analysis_store import SimpleDreamsAnalysisStore
from dreamsboard.engine.storage.storage_context import StorageContext
from dreamsboard.engine.utils import concat_dirs

langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def test_structured_dreams_storyboard_store_test7(setup_log) -> None:
    llm = ChatOpenAI(
        openai_api_base='http://127.0.0.1:30000/v1',
        model="glm-4",
        openai_api_key="glm-4",
        verbose=True
    )
    guidance_llm = ChatOpenAI(
        openai_api_base='http://127.0.0.1:30000/v1',
        model="glm-3-turbo",
        openai_api_key="glm-4",
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
    try:

        storage_context = StorageContext.from_defaults(persist_dir="./storage_ieAkpNXB_keyframe")
        code_gen_builder = load_store_from_storage(storage_context)
        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:

        try:

            dreams_analysis_store = SimpleDreamsAnalysisStore.from_persist_dir(persist_dir="./storage_ieAkpNXB_keyframe")

            dreams_analysis_store_loaded = True
        except:
            dreams_analysis_store_loaded = False

        if not dreams_analysis_store_loaded:
            dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
                llm=llm, csv_file_path="/media/gpt4-pdf-chatbot-langchain/InterpretationoDreams/src/docs/csv/ieDRHjmD_keyframe.csv")

            output = dreams_generation_chain.run()
            dreams_guidance_context = output.get("dreams_guidance_context").get("dreams_guidance_context")
            dreams_personality_context = output.get("dreams_personality_context").get("dreams_personality_context")
            # 拼接dreams_guidance_context和dreams_personality_context两个字典
            dreams_generation = {}
            dreams_generation.update(output.get("dreams_guidance_context"))
            dreams_generation.update(output.get("dreams_personality_context"))

            dreams_analysis_store = SimpleDreamsAnalysisStore()
            dreams = DreamsPersonalityNode.from_config(cfg=dreams_generation)
            dreams_analysis_store.add_analysis([dreams])
            logger.info(dreams_analysis_store.analysis_all)
            dreams_analysis_store_path = concat_dirs(dirname="./storage_ieAkpNXB_keyframe", basename="dreams_analysis_store.json")
            dreams_analysis_store.persist(persist_path=dreams_analysis_store_path)
        else:
            for val in dreams_analysis_store.analysis_all.values():
                dreams_guidance_context = val.dreams_guidance_context
                dreams_personality_context = val.dreams_personality_context

        builder = StructuredStoryboardCSVBuilder.form_builder(csv_file_path="/media/gpt4-pdf-chatbot-langchain/InterpretationoDreams/src/docs/csv/ieDRHjmD_keyframe.csv")
        builder.load()
        storyboard_executor = StructuredDreamsStoryboard.form_builder(llm=llm,
                                                                      builder=builder,
                                                                      dreams_guidance_context=dreams_guidance_context,
                                                                      dreams_personality_context=dreams_personality_context,
                                                                      guidance_llm=guidance_llm
                                                                      )
        code_gen_builder = storyboard_executor.loader_cosplay_builder(
            engine_template_render_data={
                'model_name': 'glm-4',
                'OPENAI_API_BASE': 'http://127.0.0.1:30000/v1',
                'OPENAI_API_KEY': 'glm-4',
            })

        # persist index to disk
        code_gen_builder.storage_context.dreams_analysis_store = dreams_analysis_store
        code_gen_builder.storage_context.persist(persist_dir="./storage_ieAkpNXB_keyframe")

    executor = code_gen_builder.build_executor()
    logger.info(executor)
    logger.info(executor.executor_code)


    assert True
