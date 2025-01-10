from __future__ import annotations

"""

Implements the MCTS + Self-Refine algorithm from
`Accessing GPT-4 level Mathematical Olympiad Solutions via Monte
Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report`
by Zhang et. al.

The authors' [repo](https://github.com/trotsky1997/MathBlackBox) uses critiques,
refinements, and parent nodes' answers as conversation history.
I haven't tried it yet.

"""

"""

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

"""

import random
import math
from collections import deque
from enum import Enum 
from pydantic import BaseModel
import tqdm
from dreamsboard.engine.memory.mctsr.prompt import (
    gpt_prompt_config,
    RefineResponse,
)
from dreamsboard.engine.task_engine_builder.core import CodeGeneratorBuilder
from dreamsboard.document_loaders.structured_storyboard_loader import LinkedListNode

from dreamsboard.engine.generate.code_generate import QueryProgramGenerator, EngineProgramGenerator
from langchain.schema import AIMessage
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)



ROOT_UCT_SCORE = 10_000


class MCTSNode(BaseModel):
    answer: str
    linked_list_node: LinkedListNode
    """当前任务的会话信息""" 
    code_gen_builder: CodeGeneratorBuilder
    """当前任务的会话执行者""" 
    engine_template_render_data: dict = {}
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    visits: int = 0
    Q: float = 0
    reward_samples: list[int] = []

    class Config:
        arbitrary_types_allowed = True
        
    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits})"

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)

        # Average worst-case and average outcomes
        self.Q = (min_reward + avg_reward) / 2


class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3

 


class MCTSr(BaseModel):
    problem: str
    max_rollouts: int
    exploration_constant: float = 1.0
    max_children: int = 2
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    selection_policy: SelectionPolicy = SelectionPolicy.GREEDY 

    root: MCTSNode | None = None

    # Logs
    critiques: list[str] = []
    refinements: list[str] = []
    rewards: list[float] = []
    selected_nodes: list[MCTSNode] = []

    class Config:
        arbitrary_types_allowed = True
        
    def self_refine(self, node: MCTSNode) -> MCTSNode:
        raise NotImplementedError()

    def _evaluate_answer(self, node: MCTSNode) -> int:
        raise NotImplementedError()

    def self_evaluate(self, node: MCTSNode):
        """Evaluate the quality of the answer. Sample `num_samples` times and average the results."""
        reward = self._evaluate_answer(node)

        if reward > self.reward_limit:
            reward -= self.excess_reward_penalty

        node.add_reward(reward)

    def backpropagate(self, node: MCTSNode):
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visits += 1
            parent = parent.parent

    def uct(self, node: MCTSNode):
        if not node.parent:
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth.
            return ROOT_UCT_SCORE

        return node.Q + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )

    def is_fully_expanded(self, node: MCTSNode):
        return len(node.children) >= self.max_children or any(
            child.Q > node.Q for child in node.children
        )

    def select_node(self):
        """Select a non-fully expanded node with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own
        """
        candidates: list[MCTSNode] = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        if not candidates:
            return self.root

        if self.selection_policy == SelectionPolicy.GREEDY:
            return max(candidates, key=self.uct)
        elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
            # Sample, weighted by UCT score
            uct_scores = [self.uct(node) for node in candidates]
            selected_pair_idx = random.choices(
                range(len(candidates)), weights=uct_scores, k=1
            )[0]
            return candidates[selected_pair_idx]
        elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
            # Sample, weighted by the difference in UCT scores between pairs
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [
                (i, j) for i in range(len(candidates)) for j in range(len(candidates))
            ]
            pair_weights = [
                max(uct_scores[i], uct_scores[j]) - min(uct_scores[i], uct_scores[j])
                for i, j in pairs
            ]
            selected_pair_idx = random.choices(
                range(len(pairs)), weights=pair_weights, k=1
            )[0]
            selected_candidate_idx = max(
                pairs[selected_pair_idx], key=lambda x: uct_scores[x]
            )
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")
 

    def initialize(self, root_node: MCTSNode):
        """Generate a zero-shot answer."""
        if not isinstance(root_node, MCTSNode):
            raise ValueError("root_node must be an instance of MCTSNode")
        self.root = root_node

    def run(self): 
        for _ in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            self.self_evaluate(node)
            child = self.self_refine(node)
            node.add_child(child)
            self.self_evaluate(child)
            self.backpropagate(child)

        return self.get_best_answer()

    def get_best_answer(self):
        from collections import deque

        to_visit = deque([self.root])
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)

        return best_node.answer

    def print(self):
        print_tree(self.root)



class MCTSrStoryboard(MCTSr): 

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        """
        自我反思
        """
  
        user_prompt = "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    )
        _ai_message = self._get_ai_message('system', gpt_prompt_config.critic_system_prompt, user_prompt, node)
        assert _ai_message.content is not None
        critique = _ai_message.content
        assert critique is not None
        self.critiques.append(critique)
  
        user_prompt = "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    )
        _refined_answer_response_message = self._get_ai_message('system', gpt_prompt_config.refine_system_prompt, user_prompt, node)
        assert _refined_answer_response_message.content is not None

        refined_answer = RefineResponse.model_validate_json(
            _refined_answer_response_message.content
        )
        self.refinements.append(refined_answer)

        # persist index to disk
        node.code_gen_builder.storage_context.persist(persist_dir=f"./storage/{node.linked_list_node.task_step_id}")
        
        return MCTSNode(
            answer=f"# Thought {refined_answer.thought}\n\n# Answer\n{refined_answer.answer}",
            parent=node,
        )

    def _evaluate_answer(self, node: MCTSNode) -> int:
        """
        评估答案
        """
        role = 'system'

        system_prompt = gpt_prompt_config.evaluate_system_prompt
        user_prompt = "\n\n".join(
            [
                f"<problem>\n{self.problem}\n</problem>",
                f"<answer>\n{node.answer}\n</answer>",
            ]
        )
        for attempt in range(3):
            try:
                _ai_message = self._get_ai_message(role, system_prompt, user_prompt, node, remove_context=False) 
                assert _ai_message.content is not None
                return int(_ai_message.content)
            except ValueError:
                role = 'assistant'
                system_prompt =  _ai_message.content
                user_prompt = "Failed to parse reward as an integer."
               
                if attempt == 2:
                    raise
    
    def _get_ai_message(self, role: str, system_prompt: str, user_prompt: str, node: MCTSNode, remove_context: bool = True) -> AIMessage:
        """
        获取AI消息
        """
                
        node.code_gen_builder.add_generator(QueryProgramGenerator.from_config(cfg={
            "query_code_file": "query_template.py-tpl",
            "render_data": {
                'cosplay_role': role,
                'message': system_prompt
            }
        }))

        node.code_gen_builder.add_generator(QueryProgramGenerator.from_config(cfg={
            "query_code_file": "query_template.py-tpl",
            "render_data": {
                'cosplay_role': 'user',
                'message': user_prompt
            },
        }))

        node.code_gen_builder.add_generator(EngineProgramGenerator.from_config(cfg={
            "engine_code_file": "simple_engine_template.py-tpl",
            "render_data": node.engine_template_render_data
        }))

        executor = node.code_gen_builder.build_executor()
        logger.info(executor.executor_code)

        executor.execute()
        _ai_message = executor.chat_run()

        logger.info(executor._ai_message)
        assert executor._ai_message is not None
        # 删除上下文
        if remove_context:
            node.code_gen_builder.remove_last_generator()
            node.code_gen_builder.remove_last_generator()
            node.code_gen_builder.remove_last_generator()
        else:
            node.code_gen_builder.remove_last_generator()

        return _ai_message


def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        logger.info(indent + line)
    for child in node.children:
        print_tree(child, level + 1)