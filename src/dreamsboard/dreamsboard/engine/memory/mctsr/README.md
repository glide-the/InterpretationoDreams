

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

