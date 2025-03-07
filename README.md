# InterpretationoDreams

## Description
使用langchain进行任务规划，构建子任务的会话场景资源，通过MCTS任务执行器，来让每个子任务通过在上下文中资源，通过自身反思探索来获取自身对问题的最优答案；这种方式依赖模型的对齐偏好，我们在每种偏好上设计了一个工程框架，来完成自我对不同答案的奖励进行采样策略
 

## 使用
本项目分成了不同的任务模块，

[角色扮演](src/docs/coplay_analysis/README.md)请参考`src/docs/coplay_analysis/README.md`, 


[任务规划](src/docs/task_step/README.md)请参考`src/docs/task_step/README.md` [MCTS的工程化实现+长文本思维链生成思路分享.md](src/docs/task_step/MCTS的工程化实现+长文本思维链生成思路分享.md)
 

## 正在进行中的事情

- 调整MCTS的反思策略，增加上下文窗口


    构建 StructuredStoryboard 对象，并确保生成的上下文不超过指定 token 数量。
    默认规则为：以当前 task_step_id 为中心，取其前后步骤的内容。