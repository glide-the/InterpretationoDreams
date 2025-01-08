# InterpretationoDreams

## Description
使用langchain实现 故事情景生成，情感情景引导，剧情总结，性格分析


### 数据处理方式
相关工具
- `https://github.com/glide-the/Keyframe-Extraction-for-video-summarization` 提取视频关键帧,结合字幕文件整理分镜信息
- `https://github.com/LC1332/Chat-Haruhi-Suzumiya` 提取视频字幕及其对应的时间戳
- `https://github.com/glide-the/weaviate-segm-chunk`  结合GPT与概念检索的召回策略，解决长上下文代指等问题


 
主要模块
- 个体心理状态提取思维链： 通过分析剧本和结果，实现了结构化的个体内在心理状态及其行为的驱动力。情感细节从原本的字幕中被有效提取，加载对话加载器以进行角色扮演。
- 音视频处理模块： 调用了 TransNetV2 和 whispe 音视频处理模型，封装了完整的数据加载代码，以满足后续思维链加载及数据分析的需求。
- 设计实现会话加载器： 通过对 langchain 的每个代码块进行抽象，构建了一个代码生成器，将langchai 工具分成基础程序、逻辑控制程序、逻辑加载程序、逻辑运行程序。这个设计使得在应用数据模块概念后更加实用和简单。
- 设计记忆模块： 自定义了不同形态的记忆存储形式，为后续行动决策与环境感知的任务协同分析提供了关键指标，同时增强了人类对行为的可解释性和可信度。


##### 数据整理 [README.md](src/docs/README.md)



## 近期任务

(不重要)任务一、GraphRAG 图抽取代码重构，用于自定义概念词


任务二、对任务进行规划，生成段落之间组成一个动态上下文

任务三、设计场景解析器，文档编写所需的提示词在（src/dreamsboard/tests/test_hydrology/prompts.py）


任务二 需求：

1、对任务按照提示词要求进行扩写，将扩写任务步骤收集 （src/dreamsboard/dreamsboard/engine/entity/task_step、src/dreamsboard/tests/test_kor/test_kor3.py）

2、收集每个任务后存储到磁盘（src/dreamsboard/dreamsboard/engine/storage/task_step_store）

3、对每个子任务载入会话场景，然后按照扩写任务步骤构建，MCTS任务



目标：设计满足不同场景的思维链


#### 场景加载模块

编写符合计算机科学领域的 故事情境提示词，生成研究情境（story_scenario_context），替换现有的langchain会话模板，
1、对这个提示词所要求的输入拆分成子任务，
2、对每个子任务指令转换为子问题，召回问题前3条，
3、对召回内容与问题拼接，合成会话内容变量（scene_content）


对每个子问题相关的召回内容，转换为第一人称的会话总结（研究场景（scene_monologue_context）），

1、对召回内容与问题拼接，对每个前3条组成一个总结任务的提示词，为每个任务标记唯一编号，组成任务上下文（story_board_summary_context）
2、加载编号和story_board_summary_context，转换会话信息



### MCTS任务构建

MCT 自优化算法代表了蒙特卡洛树搜索（MCTS）与大型语言模型的结合，将不同的场景执行任务过程抽象为搜索树结构。树上的节点代表当前不同视角下的选择策略，而边表示主体对自身的反思。该算法的操作流程遵循 MCTS 算法的一般模式。

具体来说，我们采用模型的先验知识，来让主体通过一系列的自身反思探索来获取自身对问题的最优答案；这种方式依赖模型的对齐偏好，我们在每种偏好上设计了一个工程框架，来完成自我对不同答案的奖励进行采样策略


1、对问题生成的子任务，生成一个合理的规划的节点
2、对每个节点创建一个MCTS任务，
3、输入 problem（总问题的子任务相关的子问题）
4、评分代码重构，将片段摘录器模块集成到一个关于_evaluate_answer逻辑提示模板，模板主要作用：将每个子问题相关的loader_cosplay_builder构建一个关于evaluate_system_prompt 的生成策略，具体的为编写一个关于带有评估的评估器，由loader_cosplay_builder方法返回场景执行器（CodeGeneratorBuilder），使用add_generator添加一个问答策略(CodeGenerator)中构成问答交互，build_executor后执行  executor.chat_run() 返回_ai_message

5、自我反思代码重构,将片段摘录器模块集成到一个关于self_refine逻辑提示模板，模板主要作用：将每个子问题相关的loader_cosplay_builder构建一个关于critic_system_prompt和refine_system_prompt的生成策略，critic_system_prompt为生成一个关于子问题相关的loader_cosplay_builder中自身不完美的评价内容，refine_system_prompt为不完美评价的思考过程和评分值。
具体的为编写一个关于带有评价的生成器和反思生成器，它们由loader_cosplay_builder方法返回场景执行器（CodeGeneratorBuilder），使用add_generator添加一个问答策略(CodeGenerator)中构成问答交互，build_executor后执行  executor.chat_run() 返回_ai_message



#### MCTS执行中的重要环节
MCTS中的约束规则如下，需要保证这些节点必须符合下面所定义的基本规则

提示约束：模型在奖励评分期间必须遵守最严格的标准。生成结果需要为JSON Response format
{
    "thought": "The thought process behind the answer.",
    "answer": "A float representing the answer to the problem."
}


高分抑制：评分节点中不存在满分反馈机制；任何超过 95 分的奖励都会按固定金额减少，以遏制过高分数。

重复采样：每次访问搜索树节点都涉及对节点奖励的重复采样，以增强自我评估的可靠性。需要注意的是，当对节点的子节点进行奖励采样时，我们也会对其父节点进行奖励采样，以增加奖励采样的样本量。



## 已完成

对任务进行规划， 
1、对这个提示词所要求的输入拆分成子任务，

1、对任务按照提示词要求进行扩写，将扩写任务步骤收集 （src/dreamsboard/dreamsboard/engine/entity/task_step、src/dreamsboard/tests/test_kor/test_kor3.py）

2、收集每个任务后存储到磁盘（src/dreamsboard/dreamsboard/engine/storage/task_step_store）

场景加载模块
 
2、对每个子任务指令转换为子问题