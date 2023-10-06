from jinja2 import Template
import logging

from dreamsboard.templates import get_template_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


def test_template() -> None:
    # 读取模板文件
    with open(get_template_path('base_template.py-tpl'), 'r') as template_file:
        base_template_content = template_file.read()

    # 创建一个Jinja2模板对象
    base_template = Template(base_template_content)

    # 定义要填充到模板中的数据
    data = {
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

    # 使用模板和数据生成代码
    base_generated_code = base_template.render(data)

    # 打印生成的代码
    logger.info(base_generated_code)
    assert True

