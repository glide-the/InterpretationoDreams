import logging

from langchain.chat_models import ChatOpenAI

from dreamsboard.dreams.builder_cosplay_code.base import StructuredDreamsStoryboard
from dreamsboard.dreams.dreams_personality_chain.base import StoryBoardDreamsGenerationChain
import langchain

from dreamsboard.engine.generate.code_generate import QueryProgramGenerator, EngineProgramGenerator, AIProgramGenerator
from dreamsboard.engine.loading import load_store_from_storage
from dreamsboard.engine.storage.storage_context import StorageContext

langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def test_structured_dreams_storyboard_store() -> None:
    try:

        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        code_gen_builder = load_store_from_storage(storage_context)
        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:
        llm = ChatOpenAI(
            verbose=True
        )

        dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
            llm=llm, csv_file_path="../../docs/csv/ieAeWyXU_keyframe.csv")

        output = dreams_generation_chain.run()
        logger.info("dreams_guidance_context:" + output.get("dreams_guidance_context"))
        logger.info("dreams_personality_context:" + output.get("dreams_personality_context"))
        dreams_guidance_context = output.get("dreams_guidance_context")
        dreams_personality_context = output.get("dreams_personality_context")

        storyboard_executor = StructuredDreamsStoryboard.form_builder(llm=llm,
                                                                      builder=dreams_generation_chain.builder,
                                                                      dreams_guidance_context=dreams_guidance_context,
                                                                      dreams_personality_context=dreams_personality_context
                                                                      )
        code_gen_builder = storyboard_executor.loader_cosplay_builder()

    _dreams_render_data = {
        'cosplay_role': '宝宝',
        'message': '兔兔，我没有跟你一起打羽毛球，你会不会不开心呀？'
    }
    code_gen_builder.add_generator(QueryProgramGenerator.from_config(cfg={
        "query_code_file": "query_template.py-tpl",
        "render_data": _dreams_render_data,
    }))
    code_gen_builder.add_generator(EngineProgramGenerator.from_config(cfg={
        "engine_code_file": "simple_engine_template.py-tpl",
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
