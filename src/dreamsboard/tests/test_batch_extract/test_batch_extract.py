import logging

from langchain.chat_models import ChatOpenAI

from dreamsboard.document_loaders import load_csv, StructuredStoryboardCSVBuilder, batch
from dreamsboard.dreams.builder_cosplay_code.base import StructuredDreamsStoryboard
from dreamsboard.dreams.dreams_personality_chain.base import StoryBoardDreamsGenerationChain
import langchain
import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
import shutil
from tqdm import tqdm
import time
import re

from dreamsboard.engine.dreams_personality.dreams_personality import DreamsPersonalityNode
from dreamsboard.engine.loading import load_store_from_storage
from dreamsboard.engine.storage.dreams_analysis_store.simple_dreams_analysis_store import SimpleDreamsAnalysisStore
from dreamsboard.engine.storage.storage_context import StorageContext
from dreamsboard.engine.utils import concat_dirs
from dreamsboard.utils import get_config_dict, get_log_file, get_timestamp_ms

langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def check_and_convert_special_characters(text):
    # 将除了中文之外的所有字符转换成\U
    converted_text = ''.join(['__' if not ('\u4e00' <= char <= '\u9fff') else char for char in text])
    return converted_text


def test_batch_extract(setup_log) -> None:

    data_folder = '/media/gpt4-pdf-chatbot-langchain/InterpretationoDreams/社会交流步骤分析/msg_extract_csv'
    save_folder = "/media/gpt4-pdf-chatbot-langchain/InterpretationoDreams/社会交流步骤分析/msg_extract_storage_2024_02_19"
    ds_path = Path(save_folder)
    if ds_path.exists() is False:
        ds_path.mkdir()
    txt_files = load_csv(data_folder)
    logger.info("获取数据，成功{}".format(len(txt_files)))
    llm = ChatOpenAI(
        openai_api_base='http://127.0.0.1:30000/v1',
        model="glm-4",
        openai_api_key="sk-4ftOuS6xQ3he1MKZD8E7BeA295D04a33A7Ad3544857e70C6",
        verbose=True
    )

    # guidance_llm = ChatOpenAI(
    #     openai_api_base='http://127.0.0.1:30000/v1',
    #     model="glm-3-turbo",
    #     openai_api_key="glm-4",
    #     verbose=True,
    #     temperature=0.1,
    #     top_p=0.9,
    # )
    guidance_llm = ChatOpenAI(
        openai_api_base='http://127.0.0.1:30000/v1',
        model="glm-3-turbo",
        openai_api_key="glm-4",
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
    personality_llm = ChatOpenAI(
        openai_api_base='http://127.0.0.1:30000/v1',
        model="glm-3-turbo",
        openai_api_key="glm-4",
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
    batch_len = 10

    # 对每一个文件进行操作
    batch_list = batch(txt_files, batch_len)
    b_unit = tqdm(enumerate(txt_files), total=len(txt_files))
    for index, batch_file in enumerate(batch_list):
        # 更新描述
        b_unit.set_description("test_batch_extract batch: {}".format(index))
        # 立即显示进度条更新结果
        b_unit.refresh()
        for filename in batch_file:
            # 使用os.path.splitext()分割文件名和扩展名
            file_name, file_extension = os.path.splitext(os.path.basename(filename))
            builder = StructuredStoryboardCSVBuilder(
                csv_file_path=filename)
            builder.load()  # 替换为你的CSV文件路径
            export_role = builder.export_role()
            logger.info(export_role)
            for role in export_role:
                try:
                    try:

                        storage_context = StorageContext.from_defaults(
                            persist_dir=f"{save_folder}/{check_and_convert_special_characters(role)}/storage_{file_name}"
                        )
                        code_gen_builder = load_store_from_storage(storage_context)
                        index_loaded = True
                        logger.info(f"load storage success:{index_loaded}")
                    except:
                        index_loaded = False
                        logger.info(f"load storage error:{index_loaded}")

                    if not index_loaded:
                        try:

                            dreams_analysis_store = SimpleDreamsAnalysisStore.from_persist_dir(
                                persist_dir=f"{save_folder}/{check_and_convert_special_characters(role)}/storage_{file_name}"
                            )

                            dreams_analysis_store_loaded = True
                        except:
                            dreams_analysis_store_loaded = False

                        if not dreams_analysis_store_loaded:
                            dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
                                llm=llm, csv_file_path=filename, user_id=role)

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
                            dreams_analysis_store_path = concat_dirs(dirname=f"{save_folder}/{check_and_convert_special_characters(role)}/storage_{file_name}",
                                                                     basename="dreams_analysis_store.json")
                            dreams_analysis_store.persist(persist_path=dreams_analysis_store_path)
                        else:
                            for val in dreams_analysis_store.analysis_all.values():
                                dreams_guidance_context = val.dreams_guidance_context
                                dreams_personality_context = val.dreams_personality_context

                        builder = StructuredStoryboardCSVBuilder.form_builder(csv_file_path=filename)
                        builder.load()
                        storyboard_executor = StructuredDreamsStoryboard.form_builder(llm=llm,
                                                                                      builder=builder,
                                                                                      dreams_guidance_context=dreams_guidance_context,
                                                                                      dreams_personality_context=dreams_personality_context,                                                                                      guidance_llm=guidance_llm,
                                                                                      personality_llm=personality_llm,
                                                                                      user_id=role
                                                                                      )
                        code_gen_builder = storyboard_executor.loader_cosplay_builder(
                            engine_template_render_data={
                                'model_name': 'glm-4',
                                'OPENAI_API_BASE': 'http://127.0.0.1:30000/v1',
                                'OPENAI_API_KEY': 'glm-4',
                            })

                        executor = code_gen_builder.build_executor()
                        logger.info(executor.executor_code)

                        # persist index to disk
                        code_gen_builder.storage_context.dreams_analysis_store = dreams_analysis_store
                        code_gen_builder.storage_context.persist(
                            persist_dir=f"{save_folder}/{check_and_convert_special_characters(role)}/storage_{file_name}")
                        logger.info(f"persist storage success")
                except Exception as e:
                    logger.error(f"解析出错role{role} in {filename} error:{e}", exc_info=True)

            # 更新进度
            b_unit.update(batch_len)
    assert True
