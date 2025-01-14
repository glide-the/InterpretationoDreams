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
from typing import Tuple
from pydantic import BaseModel
import tqdm
from dreamsboard.engine.memory.mctsr.prompt import (
    gpt_prompt_config,
    RefineResponse,
)
from dreamsboard.engine.task_engine_builder.core import CodeGeneratorBuilder
from dreamsboard.document_loaders.structured_storyboard_loader import LinkedListNode

from dreamsboard.engine.generate.code_generate import QueryProgramGenerator, BaseProgramGenerator
from langchain.schema import AIMessage
from dreamsboard.common.try_parse_json_object import try_parse_json_object
from dreamsboard.document_loaders.kor_loader import KorLoader
from dreamsboard.document_loaders.protocol.ner_protocol import TaskStepRefineNode
from langchain.prompts import PromptTemplate
from langchain_core.messages import ( 
    BaseMessage,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from dreamsboard.engine.storage.storage_context import StorageContext
import numpy as np
import logging
import re
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


PATTERN = re.compile(r"thought:\s*(.*?)\nanswer:\s*([\d.]+)", re.DOTALL)

PLAINTEXT_PATTERN = re.compile(r"```plaintext?([\s\S]*?)```[\s\S]*?answer: (\d+(\.\d+)?)", re.DOTALL)

ROOT_UCT_SCORE = 10_000


class MCTSNode(BaseModel):
    base_path: str
    answer: str
    linked_list_node: LinkedListNode
    """当前任务的会话信息""" 
    storage_context: StorageContext
    """当前任务的会话存储""" 
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
    
    llm_runable: Runnable[LanguageModelInput, BaseMessage]
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
        
    def self_refine(self, node: MCTSNode) -> Tuple[MCTSNode, RefineResponse]:
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
            if parent.children is None or len(parent.children) == 0:
                parent = parent.parent
                continue

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
            child, refined_answer = self.self_refine(node)
            node.add_child(child)
          
            if refined_answer.answer_score > self.reward_limit:
                refined_answer.answer_score -= self.excess_reward_penalty

            child.add_reward(refined_answer.answer_score)
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

    def self_refine(self, node: MCTSNode) ->  Tuple[MCTSNode, RefineResponse]:
        """
        自我反思
        """

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
        past_steps = ''
        past_context = ''
        context_linked_list_node = node.linked_list_node.head
        past_steps += f'{context_linked_list_node.task_step_level} task_id: {context_linked_list_node.task_step_id}, {context_linked_list_node.task_step_name}\n'
        past_context += f'{context_linked_list_node.task_step_level} task_id: {context_linked_list_node.task_step_id}, {context_linked_list_node.task_step_question_answer}\n'
        while context_linked_list_node is not None and context_linked_list_node.next is not None:
            if context_linked_list_node.task_step_id == node.linked_list_node.task_step_id:
                break
            context_linked_list_node = context_linked_list_node.next
            
            past_steps += f'{context_linked_list_node.task_step_level} task_id: {context_linked_list_node.task_step_id}, {context_linked_list_node.task_step_name}\n'
            past_context += f'{context_linked_list_node.task_step_level} task_id: {context_linked_list_node.task_step_id}, {context_linked_list_node.task_step_question_answer}\n'
            

        user_prompt = critic_system_prompt_template.format(
            problem=self.problem,
            current_answer=node.answer,
            context=past_context,
            past_steps=past_steps,
            start_task_context=node.linked_list_node.start_task_context,
            aemo_representation_context=node.linked_list_node.aemo_representation_context,
            task_step_name=node.linked_list_node.task_step_name,
            task_step_description=node.linked_list_node.task_step_description,
            task_step_level=node.linked_list_node.task_step_level
        ) 
        _ai_message = self._get_ai_message(gpt_prompt_config.critic_system_prompt, user_prompt, node)
        assert _ai_message.content is not None
        critique = _ai_message.content
        assert critique is not None
        self.critiques.append(critique)
   
        refine_system_prompt_template = PromptTemplate(
            input_variables=[
                "problem", 
                "current_answer",
                "critique",
                "context",
                "past_steps",
            ],
            template=os.environ.get(
                "refine_system_prompt_data", gpt_prompt_config.refine_system_prompt_data
            )
        )
        refine_system_prompt = refine_system_prompt_template.format(
            problem=self.problem,
            current_answer=node.answer,
            critique=critique,
            context=past_context,
            past_steps=past_steps
        )
 
        _refined_answer_response_message = self._get_ai_message(gpt_prompt_config.refine_system_prompt, refine_system_prompt, node)
        assert _refined_answer_response_message.content is not None
            
        json_object = {}
        try:
            task_step_refine_node_list = self._kor_task_step_refine_builder(_refined_answer_response_message)
            # 将列表answer_score平均
            answer_score_list: list[float] = []
            for task_step_refine_node in task_step_refine_node_list:
                try:
                    answer_score_list.append(float(task_step_refine_node.answer_socre))
                except ValueError:
                    answer_score_list.append(0.0)
            answer_score = sum(answer_score_list) / len(answer_score_list)
            if answer_score <= 1:
                answer_score = answer_score * 100

            json_object.update({
                "thought": task_step_refine_node_list[0].thought,
                "answer": task_step_refine_node_list[0].answer,
                "answer_score": answer_score
            })
        except Exception as e: 
            json_object = {
                "thought": "解析失败",
                "answer": node.answer,
                "answer_score": 0
            }

        logger.info("\033[1;32m" + f"解析后的 JSON 对象: {json_object}" + "\033[0m")
        refined_answer = RefineResponse.model_validate(json_object)
        self.refinements.append(refined_answer)

        # ```thought \n{refined_answer.thought} ```\n\n ```answer_score \n{refined_answer.answer_score} ```
        return MCTSNode(
            base_path=node.base_path,
            answer=f"{refined_answer.answer}",
            linked_list_node=node.linked_list_node,
            storage_context=node.storage_context,
            parent=node,
            children=[],
            visits=0,
            Q=0,
            reward_samples=[]
        ), refined_answer

    def _evaluate_answer(self, node: MCTSNode) -> int:
        """
        评估答案
        """ 
 
        evaluate_system_prompt_template = PromptTemplate(
            input_variables=[
                "problem", 
                "answer"
            ],
            template=os.environ.get(
                "evaluate_system_prompt_data", gpt_prompt_config.evaluate_system_prompt_data
            )
        )

        user_prompt = evaluate_system_prompt_template.format(
            problem=self.problem,
            answer=node.answer,
        )
        
        for attempt in range(3):
            try:
                _ai_message = self._get_ai_message(gpt_prompt_config.evaluate_system_prompt, user_prompt, node) 
                assert _ai_message.content is not None
                return int(_ai_message.content)
            except ValueError: 
                user_prompt = f"{_ai_message.content}\n\nFailed to parse reward as an integer."
               
                if attempt == 2:
                    raise
    
    def _get_ai_message(self, system_prompt: str, user_prompt: str, node: MCTSNode) -> AIMessage:
        """
        获取AI消息
        """
        
        code_gen_builder = CodeGeneratorBuilder.from_template(nodes=[], storage_context=node.storage_context)
        _base_render_data = {
            'system_prompt': system_prompt,
            'messages': [user_prompt]
        }
        code_gen_builder.add_generator(BaseProgramGenerator.from_config(cfg={
            "code_file": "base_template_system.py-tpl",
            "render_data": _base_render_data,
        }))
 
        executor = code_gen_builder.build_executor(
            llm_runable=self.llm_runable,
            messages=[]
        )
        executor.execute()
        _ai_message = executor.chat_run()

        logger.info("\033[1;32m" + f"_ai_message: {_ai_message}" + "\033[0m")
        assert executor._ai_message is not None 

        return _ai_message
    
    def _kor_task_step_refine_builder(self, refined_answer_response_message: AIMessage) -> list[TaskStepRefineNode]:
        """
        抽取根据批评意见优化当前回答并续写上下文内容
        """
        kor_task_step_refine_builder = KorLoader.form_kor_task_step_refine_builder(self.llm_runable)
        response = kor_task_step_refine_builder.run(refined_answer_response_message.content)
        task_step_refine_node_list = []
        if response.get('data') is not None and response.get('data').get('script') is not None:
            step_list = response.get('data').get('script')
            for step in step_list:
                task_step_refine_node = TaskStepRefineNode(
                                            thought=step.get('thought'),
                                            answer=step.get('answer'),
                                            answer_socre=step.get('answer_socre')
                                            )
                task_step_refine_node_list.append(task_step_refine_node)

        return task_step_refine_node_list

def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        logger.info(indent + line)
    for child in node.children:
        print_tree(child, level + 1)