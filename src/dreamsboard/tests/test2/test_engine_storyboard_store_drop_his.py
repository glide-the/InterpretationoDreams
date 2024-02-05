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


def test_structured_dreams_storyboard_store_drop_his() -> None:
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    code_gen_builder = load_store_from_storage(storage_context)


    executor = code_gen_builder.build_executor()
    logger.info("之前："+executor.executor_code)

    # 删除最后一个生成器，然后添加一个AI生成器
    code_gen_his = 3
    # 循环删除最后一个生成器
    for i in range(code_gen_his):
        code_gen_builder.remove_last_generator()

    executor = code_gen_builder.build_executor()

    logger.info("之后："+executor.executor_code)
    # persist index to disk
    code_gen_builder.storage_context.persist(persist_dir="./storage")
    assert True
