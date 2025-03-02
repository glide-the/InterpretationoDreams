## 使用
我们提供了一键运行脚本，由于使用了多线程，并不支持jupyter中运行，
### 如何运行
- 安装依赖
```
pip install dreamsboard["vector"] -U
```

我们对每个脚本提供了一些环境变量，除了基本的推理服务环境之外，还有一些资源配置的环境变量
- 服务商环境
```

export DEEPSEEK_API_BASE="https://api.deepseek.com/v1"
export DEEPSEEK_API_MODEL="deepseek-chat"
export DEEPSEEK_API_KEY="sk-api"
export ZHIPUAI_API_BASE="https://open.bigmodel.cn/api/paas/v4"
export ZHIPUAI_API_MODEL="glm-4-plus"
export ZHIPUAI_API_KEY="api.key"

```

- 资源配置
```
# rerank的模块，需要支持 from sentence_transformers import CrossEncoder
export cross_encoder_path="/mnt/ceph/develop/jiawei/model_checkpoint/jina-reranker-v2-base-multilingual"
# embedding的模块，需要支持 from sentence_transformers import SentenceTransformer
export embed_model_path="/mnt/ceph/develop/jiawei/model_checkpoint/m3e-base"
# 任务描述
export start_task_context="MCTS在PRM偏好策略模型微调的应用探索综述"
# 是否是一个新任务
export allow_init="true"
```


导入环境后，请使用如下脚本`test_task/glm/main.py`运行你需要的服务

- 推理
```
python test_task/glm/main.py
```
> 这个脚本会在执行位置创建本地目录，包含了`storage`中间过程，`vector_store`矢量库

> 这个过程会涉及大量的io处理请使用本地磁盘，网络磁盘会影响调度速度



 
### 渲染文档

我们也提供了一个默认的文档渲染封装，如果你想渲染其它形式的结构，请读取`storage`中间过程自行编写代码

```
python test_task/glm/printmd.md
```
> 脚本会读取`start_task_context`环境变量

 
### 设计思路  [MCTS的工程化实现+长文本思维链生成思路分享.md](src/docs/task_step/MCTS的工程化实现+长文本思维链生成思路分享.md)
### 模块说明

#### 任务规划模块设计方案
 
```
规划模块设计初期方案，只对问题进行了简单的任务拆分,有两个提示词完成操作，封装代码在AEMORepresentationChain
AEMO_REPRESENTATION_PROMPT_TEMPLATE
用于确定任务目标，规定任务场景，以及任务的话题，最终输出带有标题副标题子列表的简单描述文本
KorLoader.form_kor_dreams_task_step_builder
用于对文本结构化抽取，获取提取步骤的名称，提取每个步骤的具体建议和问题，提取步骤的层级编号

上下文交互最终由存储管理器BaseTaskStepStore定义两种范围（全局上下文、任务上下文）

由loader_task_step_iter_builder完成任务资源的加载
TODO 目前全局上下文并没有参与交互

```

 

#### 场景加载模块设计方案
```

编写符合计算机科学领域的 故事情境提示词，生成研究情境（story_scenario_context），替换现有的langchain会话模板，
1、对这个提示词所要求的输入拆分成子任务，
2、对每个子任务指令转换为子问题，召回问题前3条，
3、对召回内容与问题拼接，合成会话内容变量（scene_content）


对每个子问题相关的召回内容，转换为第一人称的会话总结（研究场景（scene_monologue_context）），

1、对召回内容与问题拼接，对每个前3条组成一个总结任务的提示词，为每个任务标记唯一编号，组成任务上下文（story_board_summary_context）
2、加载编号和story_board_summary_context，转换会话信息
```


#### MCTS设计方案

```
MCT 自优化算法代表了蒙特卡洛树搜索（MCTS）与大型语言模型的结合，将不同的场景执行任务过程抽象为搜索树结构。树上的节点代表当前不同视角下的选择策略，而边表示主体对自身的反思。该算法的操作流程遵循 MCTS 算法的一般模式。

具体来说，我们采用模型的先验知识，来让每个子任务通过在上下文中资源，通过自身反思探索来获取自身对问题的最优答案；这种方式依赖模型的对齐偏好，我们在每种偏好上设计了一个工程框架，来完成自我对不同答案的奖励进行采样策略


1、对问题生成的子任务，生成一个合理的规划的节点
2、对每个节点创建一个MCTS任务，
3、输入 problem（总问题的子任务相关的子问题）
4、评分代码重构，将片段摘录器模块集成到一个关于_evaluate_answer逻辑提示模板，模板主要作用：将每个子问题相关的loader_cosplay_builder构建一个关于evaluate_system_prompt 的生成策略，具体的为编写一个关于带有评估的评估器，由loader_cosplay_builder方法返回场景执行器（CodeGeneratorBuilder），使用add_generator添加一个问答策略(CodeGenerator)中构成问答交互，build_executor后执行  executor.chat_run() 返回_ai_message

5、自我反思代码重构,将片段摘录器模块集成到一个关于self_refine逻辑提示模板，模板主要作用：将每个子问题相关的loader_cosplay_builder构建一个关于critic_system_prompt和refine_system_prompt的生成策略，critic_system_prompt为生成一个关于子问题相关的loader_cosplay_builder中自身不完美的评价内容，refine_system_prompt为不完美评价的思考过程和评分值。
具体的为编写一个关于带有评价的生成器和反思生成器，它们由loader_cosplay_builder方法返回场景执行器（CodeGeneratorBuilder），使用add_generator添加一个问答策略(CodeGenerator)中构成问答交互，build_executor后执行  executor.chat_run() 返回_ai_message
```



#### MCTS执行中的重要环节
```
MCTS中的约束规则如下，需要保证这些节点必须符合下面所定义的基本规则

提示约束：模型在奖励评分期间必须遵守最严格的标准。生成结果需要为JSON Response format
{
    "thought": "The thought process behind the answer.",
    "answer": "A float representing the answer to the problem."
}


高分抑制：评分节点中不存在满分反馈机制；任何超过 95 分的奖励都会按固定金额减少，以遏制过高分数。

重复采样：每次访问搜索树节点都涉及对节点奖励的重复采样，以增强自我评估的可靠性。需要注意的是，当对节点的子节点进行奖励采样时，我们也会对其父节点进行奖励采样，以增加奖励采样的样本量。
```



## 调用说明
```
StructuredTaskStepStoryboard
    
对任务进行规划，生成段落之间组成一个动态上下文，  扩写任务步骤构建MCTS任务
    输入：
    	start_task_context： 起始任务
    	llm： 模型
    	kor_dreams_task_step_llm: 任务抽取模型
    	task_step_store: 任务存储器（SimpleTaskStepStore）
    	cross_encoder_path: ranking模型路径（sentence_transformers），当前设计耦合了业务，之后会单独设计一个召回模块

    任务：

        1、对任务（AEMO_REPRESENTATION_PROMPT_TEMPLATE）按照提示词要求进行扩写，将扩写任务步骤收集 （src/dreamsboard/dreamsboard/engine/entity/task_step、src/dreamsboard/tests/test_kor/test_kor3.py）

        2、收集每个任务后存储到磁盘（src/dreamsboard/dreamsboard/engine/storage/task_step_store）

        3、对每个子任务载入会话场景，然后按照扩写任务步骤构建，MCTS任务 loader_task_step_iter_builder


	》 TaskEngineBuilder 场景加载模块
		执行会话场景资源初始化，构建MCTS任务

    根据任务步骤，构建场景加载模块，生成资源文件csv
    根据每个任务，载入StructuredDreamsStoryboard 会话场景
    按照扩写任务步骤构建MCTS任务

		输入：
			task_step_id
			task_step_store: 任务存储器（SimpleTaskStepStore）
			start_task_context： 起始任务
			llm： 模型

		任务：
			init_task_engine：
				初始化任务引擎
        》TaskStepToQuestionChain 
        	输入：
        		client： 矢量库客户端
        		llm： 模型
					invoke_task_step_to_question：1、 对开始任务进行抽取，得到任务步骤，提示词所要求的输入拆分成子任务， 
					invoke_task_step_question_context： 2、对每个子任务指令转换为子问题，召回问题前3条，对任务步骤进行抽取，得到任务步骤的上下文
					export_csv_file_path: 3、对召回内容与问题 导出csv文件

			init_task_engine_dreams
				初始化场景加载资源
					StoryBoardDreamsGenerationChain
					对每个子任务通过职业提示词，载入会话场景
						1、构建场景信息（story_scenario_context），提示词（STORY_BOARD_SCENE_TEMPLATE）
						2、对任务上下文(story_board_summary_context)，构建第一人称数据(scene_monologue_context),提示词（STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE）
						3、对任务上下文(story_board_summary_context)，获取任务分析(evolutionary_step), 提示词（EDREAMS_EVOLUTIONARY_TEMPLATE）
						4、对任务分析(evolutionary_step)，分析对话预设信息（性格）， 提示词（EDREAMS_PERSONALITY_TEMPLATE）
						5、对任务上下文(story_board_summary_context)，场景信息story_scenario_context, 第一人称数据(scene_monologue_context)，
						生成关于人物职业的引导话术，提示词（DREAMS_GEN_TEMPLATE）

			init_task_engine_storyboard_executor

				构建会话场景执行器 StructuredDreamsStoryboard
					对剧本和分析结果进行结构化，将开放问题与性格分析结果进行结合。生成情景扮演会话场景执行器
			    此过程如下
			        对开放问题结果进行抽取，得到问题内容
			        对性格分析结果进行抽取，得到性格分析结果
			        增加系统提示词
			        根据剧本与任务性格基础情景扮演代码，根据每步的抽取析得到的问题，生成问题的答案
			       	在上下文中增加，关于人物职业的引导话术
	        		导出会话场景执行器
					
	    storyboard_code_gen_builder
	    	构建会话场景执行器

	    generate_step_answer
	    	获取主进程子任务的答案
	    
	    get_mcts_node
	    	构建MCTS树, 初始化当前任务相关的MCTS节点，并返回MCTS执行器

```
	 
 