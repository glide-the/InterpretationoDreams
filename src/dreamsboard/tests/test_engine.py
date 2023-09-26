import logging

from dreamsboard.generate.code_generate import BaseProgramGenerator, EngineProgramGenerator
from dreamsboard.generate.run_generate import CodeGeneratorBuilder

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
    builder = CodeGeneratorBuilder()
    builder.add_generator(BaseProgramGenerator.from_config(cfg={
        "code_file": "base_template.py-tpl"
    }))
    builder.add_generator(EngineProgramGenerator.from_config(cfg={
        "dreams_query_code_file": "dreams_query_template.py-tpl",
        "engine_code_file": "engine_template.py-tpl",
    }))

    # 定义要填充到模板中的数据
    render_data = {
        'dreams_cosplay_role': '心理咨询工作者',
        'dreams_message': '我听到你今天经历了一些有趣的事情，而且你似乎充满了好奇和喜悦。在这一切之中，有没有让你感到困惑或者需要探讨的问题？',
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

    executor = builder.build_executor(render_data)
    executor.execute()
    executor.chat_run()
    logger.info(executor._messages)
    logger.info(executor._ai_message)
