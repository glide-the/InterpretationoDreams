import logging

from langchain_community.chat_models import ChatOpenAI
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


def test_structured_dreams_storyboard_store_test6(setup_log) -> None:
    llm = ChatOpenAI(
        openai_api_base='http://127.0.0.1:20000/deepseek/v1',
        model="deepseek-chat",
        openai_api_key="deepseek-chat",
        verbose=True
    )
    guidance_llm = ChatOpenAI(
        openai_api_base='http://127.0.0.1:20000/deepseek/v1',
        model="glm-3-turbo",
        openai_api_key="deepseek-chat",
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
    try:

        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        code_gen_builder = load_store_from_storage(storage_context)
        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:

        try:

            dreams_analysis_store = SimpleDreamsAnalysisStore.from_persist_dir(persist_dir="./storage")

            dreams_analysis_store_loaded = True
        except:
            dreams_analysis_store_loaded = False

        if not dreams_analysis_store_loaded:
            dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
                llm=llm, csv_file_path="../../../docs/csv/iRMa9DMW_keyframe.csv")

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
            dreams_analysis_store_path = concat_dirs(dirname="./storage", basename="dreams_analysis_store.json")
            dreams_analysis_store.persist(persist_path=dreams_analysis_store_path)
        else:
            for val in dreams_analysis_store.analysis_all.values():
                dreams_guidance_context = val.dreams_guidance_context
                dreams_personality_context = val.dreams_personality_context

        builder = StructuredStoryboardCSVBuilder.form_builder(csv_file_path="../../../docs/csv/iRMa9DMW_keyframe.csv")
        builder.load()
        storyboard_executor = StructuredDreamsStoryboard.form_builder(llm=llm,
                                                                      builder=builder,
                                                                      dreams_guidance_context=dreams_guidance_context,
                                                                      dreams_personality_context=dreams_personality_context,
                                                                      guidance_llm=guidance_llm
                                                                      )
        code_gen_builder = storyboard_executor.loader_cosplay_builder(
            engine_template_render_data={
                'model_name': 'deepseek-chat',
                'OPENAI_API_BASE': 'http://127.0.0.1:20000/deepseek/v1',
                'OPENAI_API_KEY': 'deepseek-chat',
            })

        # persist index to disk
        code_gen_builder.storage_context.dreams_analysis_store = dreams_analysis_store
        code_gen_builder.storage_context.persist(persist_dir="./storage_ieAkpNXB_keyframe")


    _dreams_render_data = {
        'cosplay_role': '心理咨询工作者',
        'message': '''五一假期
        宝宝去了北京的北海公园，路过了鼓楼看了一个爵士乐与中国戏曲结合的演出
        你尝试下用你之前的语气，给宝宝报备一个一模一样的生活，让对方感受到你的生活，
        然后再给对方一个反馈，看看对方的反应。'''
    }
    code_gen_builder.add_generator(QueryProgramGenerator.from_config(cfg={
        "query_code_file": "query_template.py-tpl",
        "render_data": _dreams_render_data,
    }))

    code_gen_builder.add_generator(EngineProgramGenerator.from_config(cfg={
        "engine_code_file": "simple_engine_template.py-tpl",
        "render_data": {
            'model_name': 'deepseek-chat',
            'OPENAI_API_BASE': 'http://127.0.0.1:20000/deepseek/v1',
            'OPENAI_API_KEY': 'glm',
        },
    }))

    executor = code_gen_builder.build_executor()
    logger.info(executor)
    logger.info(executor.executor_code)

    assert executor.executor_code is not None
    executor.execute()
    _ai_message = executor.chat_run()

    logger.info(executor._messages)
    logger.info(executor._ai_message)
    assert executor._ai_message is not None
    # 删除最后一个生成器，然后添加一个AI生成器
    code_gen_builder.remove_last_generator()
    _ai_render_data = {
        'ai_message_content': _ai_message.content
    }
    code_gen_builder.add_generator(AIProgramGenerator.from_config(cfg={
        "ai_code_file": "ai_template.py-tpl",
        "render_data": _ai_render_data,
    }))

    # persist index to disk
    code_gen_builder.storage_context.persist(persist_dir="./storage")

    assert True


