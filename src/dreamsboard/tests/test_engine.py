import logging

from dreamsboard.engine.engine_builder import CodeGeneratorBuilder
from dreamsboard.engine.generate.code_generate import (
    CodeGenerator,
    BaseProgramGenerator,
    QueryProgramGenerator,
    AIProgramGenerator,
    EngineProgramGenerator,
)
from dreamsboard.engine.memory.mctsr.prompt import (
    gpt_prompt_config,
    RefineResponse,
)

from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import langchain
import json
import os
langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)

"""
使用设计模式，构建一个代码生成器，功能如下，
程序动态的生成python代码，python代码有三种形式，1、基础程序。2、逻辑控制程序。3、逻辑加载程序。4、逻辑运行程序

程序的执行流程如下：
1、
构建出的代码通过exec函数运行

2、
构建出的代码最终通过建造者生成一个执行器

3、
建造者在执行CodeGenerator的时候，需要使用责任链实现
"""


def test_engine() -> None:
    code_gen_builder = CodeGeneratorBuilder.from_template(nodes=[])


    _base_render_data = {
        'cosplay_role': '兔兔没有牙',
        'personality': '包括充满好奇心、善于分析和有广泛研究兴趣的人。',
        'messages': ['兔兔没有牙:「 今天是温柔长裙风。」',
                     '兔兔没有牙:「 宝宝,你再不来我家找我玩的话,这些花就全部凋谢了,你就看不到哦。」',
                     '兔兔没有牙:「 宝宝,你陪着我，我们去做一件大胆的事情。」',
                     '兔兔没有牙:「 我已经忍了很久了，我真的不想再吃丝瓜了，这根怎么又熟了，我要把它藏起来，这样大家就不知道了，他们为什么还要看花啊，那就别怪我辣手摧花吧，嘻嘻。」',
                     '兔兔没有牙:「 宝宝你看，这个小狗走路怎么还是外八，好可爱，宝宝,我弟弟给了我三颗糖，这真的能吃吗，我要吓死了,宝宝救命，小肚小肚,我在。」',
                     '兔兔没有牙:「 宝宝,我给你剥了虾,你要全部吃掉哦,乖乖.」',
                     '兔兔没有牙:「 宝宝,你想不想知道小鱼都在说什么,我来告诉你吧.」']
    }
    code_gen_builder.add_generator(BaseProgramGenerator.from_config(cfg={
        "code_file": "base_template.py-tpl",
        "render_data": _base_render_data,
    }))


    _dreams_render_data = {
        'dreams_cosplay_role': '心理咨询工作者',
        'dreams_message': '我听到你今天经历了一些有趣的事情，而且你似乎充满了好奇和喜悦。在这一切之中，有没有让你感到困惑或者需要探讨的问题？',
    }
    code_gen_builder.add_generator(QueryProgramGenerator.from_config(cfg={
        "query_code_file": "dreams_query_template.py-tpl",
        "render_data": _dreams_render_data,
    }))
    code_gen_builder.add_generator(EngineProgramGenerator.from_config(cfg={
        "engine_code_file": "engine_template.py-tpl",
    })) 
    assert True


def test_code():
    llm = ChatOpenAI(

        openai_api_base='https://open.bigmodel.cn/api/paas/v4/',
        model="glm-4-airx",
        openai_api_key='12',
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
    critic_system_prompt_template =  PromptTemplate(
        input_variables=[
            "problem", 
            "current_answer",
            "context",
            "past_steps",
            "start_task_context",
            "aemo_representation_context",
            "task_step_name",
            "task_step_description",
            "task_step_level"
        ],
        template=os.environ.get(
            "critic_system_prompt_data", gpt_prompt_config.critic_system_prompt_data 
        )
    )
   

    user_prompt = critic_system_prompt_template.format(
        problem="self.problem",
        current_answer="node.answer",
        context="past_context",
        past_steps="past_steps",
        start_task_context="node.linked_list_node.start_task_context",
        aemo_representation_context="node.linked_list_node.aemo_representation_context",
        task_step_name="node.linked_list_node.task_step_name",
        task_step_description="node.linked_list_node.task_step_description",
        task_step_level="node.linked_list_node.task_step_level"
    ) 
    code_gen_builder = CodeGeneratorBuilder.from_template(nodes=[])
    _base_render_data = {
        'system_prompt': gpt_prompt_config.critic_system_prompt,
        'messages': [user_prompt]
    }
    code_gen_builder.add_generator(BaseProgramGenerator.from_config(cfg={
        "code_file": "base_template_system.py-tpl",
        "render_data": _base_render_data,
    }))

    executor = code_gen_builder.build_executor(
        llm_runable=llm,
        messages=[]
    )

    logger.info(executor.executor_code)


def test_code1():
    
    from langchain_community.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )

    messages = []
    messages.append(SystemMessage(content = r"""完成你的目标任务,输出详细且有建设性的批评意见以改进`<current_answer>`， step by step plan. 

# 补充指南  

- 不要重复`<problem>`描述。  
- 不要重复`<current_answer>`描述。
- 不要重复`<start_task_context>`描述。
- 不要重复`<aemo_representation_context>`描述。
- 不要重复`<task_step_name>`描述。
- 不要重复`<task_step_description>`描述。
- 不要重复`<task_step_level>`描述。


结合开始任务（start_task_context），在符合任务总体描述（aemo_representation_context）的情况下，根据任务步骤名称（task_step_name）、任务步骤描述（task_step_description）和任务步骤层级（task_step_level），

突出显示需要改进或更正的特定区域。不需要更多步骤, 不要将之前完成的步骤作为计划的一部分返回。
"""))


    messages.append(HumanMessage(content = r'''你的目标:
<problem>
self.problem
</problem>

你目前的结果在这里:
<context>
past_context
</context>

<current_answer>
node.answer
</current_answer>

你目前已完成以下步骤：
past_steps


# 参考资源

start_task_context: node.linked_list_node.start_task_context
aemo_representation_context: node.linked_list_node.aemo_representation_context

# 当前任务信息

task_step_name: node.linked_list_node.task_step_name
task_step_description: node.linked_list_node.task_step_description
task_step_level: node.linked_list_node.task_step_level
'''))
    message_dicts = [convert_message_to_dict(m) for m in messages]

    logger.info(json.dumps(message_dicts,  indent=4, ensure_ascii=False))