import logging

from langchain_community.chat_models import ChatOpenAI

from dreamsboard.document_loaders import StructuredStoryboardCSVBuilder
from dreamsboard.dreams.builder_cosplay_code.base import StructuredDreamsStoryboard
from dreamsboard.dreams.dreams_personality_chain.base import StoryBoardDreamsGenerationChain
import langchain

from dreamsboard.engine.entity.dreams_personality.dreams_personality import DreamsPersonalityNode
from dreamsboard.engine.generate.code_generate import QueryProgramGenerator, EngineProgramGenerator, AIProgramGenerator
from dreamsboard.engine.loading import load_store_from_storage
from dreamsboard.engine.storage.dreams_analysis_store.simple_dreams_analysis_store import SimpleDreamsAnalysisStore
from dreamsboard.engine.storage.storage_context import StorageContext
from dreamsboard.engine.utils import concat_dirs
import os
langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def test_structured_dreams_storyboard_store_test1(setup_log) -> None:
 
    os.environ["ZHIPUAI_API_KEY"] = "5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN"
    
    os.environ["OPENAI_API_KEY"] = os.environ.get("ZHIPUAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    guidance_llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
    try:

        storage_context = StorageContext.from_defaults(persist_dir="./storage_iRMa9DMW_keyframe")
        code_gen_builder = load_store_from_storage(storage_context)
        index_loaded = True
    except:
        index_loaded = False
 


    if not index_loaded:
            
        dreams_analysis_store = SimpleDreamsAnalysisStore.from_persist_dir(persist_dir="./storage_iRMa9DMW_keyframe")


        dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
            llm=llm, csv_file_path="src/docs/csv/id46Bv3g_keyframe.csv")

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
        dreams_analysis_store_path = concat_dirs(dirname="./storage_iRMa9DMW_keyframe", basename="dreams_analysis_store.json")
        dreams_analysis_store.persist(persist_path=dreams_analysis_store_path)
 

        builder = StructuredStoryboardCSVBuilder.form_builder(csv_file_path="src/docs/csv/id46Bv3g_keyframe.csv")
        builder.load()
        storyboard_executor = StructuredDreamsStoryboard.form_builder(llm=llm,
                                                                      builder=builder,
                                                                      dreams_guidance_context=dreams_guidance_context,
                                                                      dreams_personality_context=dreams_personality_context,
                                                                      guidance_llm=guidance_llm
                                                                      )
        code_gen_builder = storyboard_executor.loader_cosplay_builder()

        # persist index to disk
        code_gen_builder.storage_context.dreams_analysis_store = dreams_analysis_store
        code_gen_builder.storage_context.persist(persist_dir="./storage_iRMa9DMW_keyframe")


    _dreams_render_data = {
        'cosplay_role': '心理咨询工作者',
        'message': '''今天是元旦跨年夜，宝宝今天没能跟你在一起。
        他准备了鸡尾酒、苹果、砂糖橘、在电脑桌前面摆的满满当当与你分享这些有趣的事情
        你尝试下用你之前的语气，给宝宝报备一个一模一样的生活，让对方感受到你的生活，
        然后再给对方一个反馈，看看对方的反应。'''
    }
    code_gen_builder.add_generator(QueryProgramGenerator.from_config(cfg={
        "query_code_file": "query_template.py-tpl",
        "render_data": _dreams_render_data,
    }))
 
    executor = code_gen_builder.build_executor(
        chat_function=llm,
        messages=[]
    )
    logger.info(executor)
    logger.info(executor.executor_code)

    assert executor.executor_code is not None
    executor.execute()
    _ai_message = executor.chat_run()

    logger.info(executor._messages)
    logger.info(executor._ai_message)
    assert executor._ai_message is not None
    # 添加一个AI生成器
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
