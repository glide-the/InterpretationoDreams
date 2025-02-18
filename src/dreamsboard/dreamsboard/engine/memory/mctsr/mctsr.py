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

import logging
import math
import os
import random
import re
import threading
from collections import deque
from enum import Enum
from typing import Tuple

import numpy as np
import tqdm
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from dreamsboard.common.callback import call_func
from dreamsboard.common.try_parse_json_object import try_parse_json_object
from dreamsboard.document_loaders.kor_loader import KorLoader
from dreamsboard.document_loaders.protocol.ner_protocol import TaskStepRefineNode
from dreamsboard.document_loaders.structured_storyboard_loader import (
    LinkedListNode,
    StructuredStoryboard,
)
from dreamsboard.dreams.task_step_md.prompts import (
    TASK_MD_TEMPLATE,
    TASK_STEP_MD_DESC_TEMPLATE,
    TASK_STEP_MD_LIST_TEMPLATE,
    TASK_STEP_MD_TEMPLATE,
    TASK_STEP_MD_TITLE_TEMPLATE,
)
from dreamsboard.engine.generate.code_generate import (
    BaseProgramGenerator,
    QueryProgramGenerator,
)
from dreamsboard.engine.memory.mctsr.prompt import (
    RefineResponse,
    gpt_prompt_config,
)
from dreamsboard.engine.storage.storage_context import StorageContext
from dreamsboard.engine.storage.task_step_store.types import BaseTaskStepStore
from dreamsboard.engine.task_engine_builder.core import CodeGeneratorBuilder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


PATTERN = re.compile(r"thought:\s*(.*?)\nanswer:\s*([\d.]+)", re.DOTALL)

PLAINTEXT_PATTERN = re.compile(
    r"```plaintext?([\s\S]*?)```[\s\S]*?answer: (\d+(\.\d+)?)", re.DOTALL
)

_PROMPT_TEMPLATE_1 = PromptTemplate(
    input_variables=[
        "task_step_name",
        "task_step_level",
        "task_step_id",
        "task_step_question_answer",
    ],
    template=TASK_STEP_MD_TITLE_TEMPLATE,
)
_PROMPT_TEMPLATE_1_1 = PromptTemplate(
    input_variables=[
        "task_step_name",
        "task_step_level",
        "task_step_id",
        "task_step_question_answer",
    ],
    template=TASK_STEP_MD_DESC_TEMPLATE,
)
_PROMPT_TEMPLATE_1_2 = PromptTemplate(
    input_variables=[
        "task_step_name",
        "task_step_level",
        "task_step_id",
        "task_step_question_answer",
    ],
    template=TASK_STEP_MD_LIST_TEMPLATE,
)
_PROMPT_TEMPLATE_1_3 = PromptTemplate(
    input_variables=[
        "task_step_name",
        "task_step_level",
        "task_step_id",
        "task_step_question_answer",
    ],
    template=TASK_STEP_MD_TEMPLATE,
)
_PROMPT_TEMPLATE_2 = PromptTemplate(
    input_variables=[
        "start_task_context",
        "aemo_representation_context",
        "context_placeholder",
    ],
    template=TASK_MD_TEMPLATE,
)
ROOT_UCT_SCORE = 10_000


class MCTSNode(BaseModel):
    task_step_id: str
    answer: str
    """当前任务的会话信息"""
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
    base_path: str
    task_step_id: str
    """当前任务的id"""
    storage_context: StorageContext
    """当前任务的会话存储"""
    llm_runable: Runnable[LanguageModelInput, BaseMessage]
    kor_dreams_task_step_llm: Runnable[LanguageModelInput, BaseMessage]
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

    def initialize(self):
        """Generate a zero-shot answer."""
        # 构建MCTS树,
        structured_storyboard = _build_structured_storyboard(
            self.storage_context.task_step_store
        )

        linked_list_node = structured_storyboard.get_task_step_node(self.task_step_id)
        #  初始化当前节点self.root的上级节点
        mcts_tree = linked_list_to_tree(structured_storyboard.head)

        matching_node = find_matching_node_in_tree_iterative(
            mcts_tree, linked_list_node
        )
        self.root = matching_node

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

        owner = f"thread {threading.get_native_id()}, run end"
        logger.info(f"owner:{owner}")
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
    """

    resource_id=f"resource_{node.task_step_id}"
    """

    @staticmethod
    def _get_ai_message(callback, resource_id, **kwargs):
        code_gen_builder = CodeGeneratorBuilder.from_template(
            nodes=[], storage_context=kwargs.get("storage_context")
        )
        _base_render_data = {
            "system_prompt": kwargs.get("system_prompt"),
            "messages": [kwargs.get("user_prompt")],
        }
        code_gen_builder.add_generator(
            BaseProgramGenerator.from_config(
                cfg={
                    "code_file": "base_template_system.py-tpl",
                    "render_data": _base_render_data,
                }
            )
        )

        executor = code_gen_builder.build_executor(
            llm_runable=kwargs.get("llm_runable"), messages=[]
        )
        executor.execute()
        _ai_message = executor.chat_run()

        logger.info("\033[1;32m" + f"{resource_id}: {_ai_message}" + "\033[0m")
        assert executor._ai_message is not None

        callback(_ai_message)

    @staticmethod
    def _wrapper_steps_unit(
        context_linked_list_node: LinkedListNode, continue_task_step_id: str
    ):
        # 使用 TASK_STEP_MD_TEMPLATE 格式化每个任务步骤
        formatted_task_steps = []
        past_steps = []

        past_steps.append(
            f"{context_linked_list_node.task_step_level} task_id: {context_linked_list_node.task_step_id}, {context_linked_list_node.task_step_name}\n"
        )

        # 计算层级关系
        level_count = context_linked_list_node.task_step_level.count(">")

        if level_count == 0:
            # 一级，格式化为标题 #
            step_text = _PROMPT_TEMPLATE_1.format(
                task_step_name=f"### {context_linked_list_node.task_step_name}",
                task_step_level=context_linked_list_node.task_step_level,
                task_step_description=context_linked_list_node.task_step_description,
                task_step_id=context_linked_list_node.task_step_id,
                task_step_question_answer=context_linked_list_node.task_step_question_answer,
            )
            formatted_task_steps.append(step_text.strip() + "\n\n")
        elif level_count == 1:
            # 二级，格式化为标题 ##
            step_text = _PROMPT_TEMPLATE_1_1.format(
                task_step_name=f"{context_linked_list_node.task_step_name}",
                task_step_level=context_linked_list_node.task_step_level,
                task_step_description=context_linked_list_node.task_step_description,
                task_step_id=context_linked_list_node.task_step_id,
                task_step_question_answer=context_linked_list_node.task_step_question_answer,
            )
            formatted_task_steps.append(step_text.strip() + "\n\n")

        elif level_count >= 2:
            # 三级及以上，格式化为分类 -
            step_text = _PROMPT_TEMPLATE_1_2.format(
                task_step_name=f"- {context_linked_list_node.task_step_name}",
                task_step_level=context_linked_list_node.task_step_level,
                task_step_description=context_linked_list_node.task_step_description,
                task_step_id=context_linked_list_node.task_step_id,
                task_step_question_answer=context_linked_list_node.task_step_question_answer,
            )
            formatted_task_steps.append(step_text.strip() + "\n\n")

        else:
            step_text = _PROMPT_TEMPLATE_1_3.format(
                task_step_name=f"{context_linked_list_node.task_step_name}",
                task_step_level=context_linked_list_node.task_step_level,
                task_step_description=context_linked_list_node.task_step_description,
                task_step_id=context_linked_list_node.task_step_id,
                task_step_question_answer=context_linked_list_node.task_step_question_answer,
            )

            formatted_task_steps.append(step_text.strip())

        while (
            context_linked_list_node is not None
            and context_linked_list_node.next is not None
        ):
            if context_linked_list_node.task_step_id == continue_task_step_id:
                break
            context_linked_list_node = context_linked_list_node.next
            # 计算层级关系
            level_count = context_linked_list_node.task_step_level.count(">")

            past_steps.append(
                f"{context_linked_list_node.task_step_level} task_id: {context_linked_list_node.task_step_id}, {context_linked_list_node.task_step_name}\n"
            )

            if level_count == 0:
                # 一级，格式化为标题 #
                step_text = _PROMPT_TEMPLATE_1.format(
                    task_step_name=f"### {context_linked_list_node.task_step_name}",
                    task_step_level=context_linked_list_node.task_step_level,
                    task_step_description=context_linked_list_node.task_step_description,
                    task_step_id=context_linked_list_node.task_step_id,
                    task_step_question_answer=context_linked_list_node.task_step_question_answer,
                )
                formatted_task_steps.append(step_text.strip() + "\n\n")

            elif level_count == 1:
                # 二级，格式化为标题 ##
                step_text = _PROMPT_TEMPLATE_1_1.format(
                    task_step_name=f"{context_linked_list_node.task_step_name}",
                    task_step_level=context_linked_list_node.task_step_level,
                    task_step_description=context_linked_list_node.task_step_description,
                    task_step_id=context_linked_list_node.task_step_id,
                    task_step_question_answer=context_linked_list_node.task_step_question_answer,
                )
                formatted_task_steps.append(step_text.strip() + "\n\n")

            elif level_count >= 2:
                # 三级及以上，格式化为分类 -
                step_text = _PROMPT_TEMPLATE_1_2.format(
                    task_step_name=f"- {context_linked_list_node.task_step_name}",
                    task_step_level=context_linked_list_node.task_step_level,
                    task_step_description=context_linked_list_node.task_step_description,
                    task_step_id=context_linked_list_node.task_step_id,
                    task_step_question_answer=context_linked_list_node.task_step_question_answer,
                )
                formatted_task_steps.append(step_text.strip() + "\n\n")

            else:
                step_text = _PROMPT_TEMPLATE_1_3.format(
                    task_step_name=f"{context_linked_list_node.task_step_name}",
                    task_step_level=context_linked_list_node.task_step_level,
                    task_step_description=context_linked_list_node.task_step_description,
                    task_step_id=context_linked_list_node.task_step_id,
                    task_step_question_answer=context_linked_list_node.task_step_question_answer,
                )

                formatted_task_steps.append(step_text.strip())

        # 将格式化的步骤列表转换为字符串
        context_placeholder = "".join(formatted_task_steps)
        past_steps_placeholder = "".join(past_steps)
        return context_placeholder, past_steps_placeholder

    def self_refine(self, node: MCTSNode) -> Tuple[MCTSNode, RefineResponse]:
        """
        自我反思
        """

        critic_system_prompt_template = PromptTemplate(
            input_variables=[
                "problem",
                "current_answer",
                "context",
                "past_steps",
                "start_task_context",
                "aemo_representation_context",
                "task_step_name",
                "task_step_description",
                "task_step_level",
            ],
            template=os.environ.get(
                "critic_system_prompt_data", gpt_prompt_config.critic_system_prompt_data
            ),
        )

        structured_storyboard = _build_structured_storyboard(
            self.storage_context.task_step_store
        )

        context_linked_list_node = structured_storyboard.head
        past_context, past_steps = self._wrapper_steps_unit(
            context_linked_list_node, node.task_step_id
        )
        task_step_node = self.storage_context.task_step_store.get_task_step(
            self.task_step_id
        )
        user_prompt = critic_system_prompt_template.format(
            problem=self.problem,
            current_answer=node.answer,
            context=past_context,
            past_steps=past_steps,
            start_task_context=task_step_node.start_task_context,
            aemo_representation_context=task_step_node.aemo_representation_context,
            task_step_name=task_step_node.task_step_name,
            task_step_description=task_step_node.task_step_description,
            task_step_level=task_step_node.task_step_level,
        )

        owner = f"register_event thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}")

        results = call_func(
            self._get_ai_message,
            resource_id=f"resource_critic_{self.task_step_id}",
            kwargs={
                "llm_runable": self.llm_runable,
                "system_prompt": gpt_prompt_config.critic_system_prompt,
                "user_prompt": user_prompt,
                "storage_context": self.storage_context,
            },
        )

        _ai_message = results[0]
        assert _ai_message.content is not None
        cleaned_text = re.sub(r'◁think▷.*?◁/think▷', '',_ai_message.content, flags=re.DOTALL)
        critique = cleaned_text
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
            ),
        )
        refine_system_prompt = refine_system_prompt_template.format(
            problem=self.problem,
            current_answer=node.answer,
            critique=critique,
            context=past_context,
            past_steps=past_steps,
        )

        owner = f"register_event thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}")

        results = call_func(
            self._get_ai_message,
            resource_id=f"resource_refine_{node.task_step_id}",
            kwargs={
                "llm_runable": self.llm_runable,
                "system_prompt": gpt_prompt_config.refine_system_prompt,
                "user_prompt": refine_system_prompt,
                "storage_context": self.storage_context,
            },
        )

        _refined_answer_response_message = results[0]
        assert _refined_answer_response_message.content is not None

        json_object = {}
        try:
            task_step_refine_node_list = self._kor_task_step_refine_builder(
                _refined_answer_response_message
            )
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

            json_object.update(
                {
                    "thought": task_step_refine_node_list[0].thought,
                    "answer": task_step_refine_node_list[0].answer,
                    "answer_score": answer_score,
                }
            )
        except Exception as e:
            json_object = {
                "thought": "解析失败",
                "answer": node.answer,
                "answer_score": 0,
            }

        logger.info("\033[1;32m" + f"解析后的 JSON 对象: {json_object}" + "\033[0m")
        refined_answer = RefineResponse.model_validate(json_object)
        self.refinements.append(refined_answer)

        # ```thought \n{refined_answer.thought} ```\n\n ```answer_score \n{refined_answer.answer_score} ```
        return MCTSNode(
            answer=f"{refined_answer.answer}",
            task_step_id=self.task_step_id,
            parent=node,
            children=[],
            visits=0,
            Q=0,
            reward_samples=[],
        ), refined_answer

    def _evaluate_answer(self, node: MCTSNode) -> int:
        """
        评估答案
        """

        evaluate_system_prompt_template = PromptTemplate(
            input_variables=["problem", "past_steps", "context", "answer"],
            template=os.environ.get(
                "evaluate_system_prompt_data",
                gpt_prompt_config.evaluate_system_prompt_data,
            ),
        )
        structured_storyboard = _build_structured_storyboard(
            self.storage_context.task_step_store
        )

        context_linked_list_node = structured_storyboard.head
        past_context, past_steps = self._wrapper_steps_unit(
            context_linked_list_node, node.task_step_id
        )

        user_prompt = evaluate_system_prompt_template.format(
            problem=self.problem,
            answer=node.answer,
            context=past_context,
            past_steps=past_steps,
        )

        for attempt in range(3):
            try:
                owner = f"register_event thread {threading.get_native_id()}, _evaluate_answer"
                logger.info(f"owner:{owner}")

                results = call_func(
                    self._get_ai_message,
                    resource_id=f"resource_evaluate_{node.task_step_id}",
                    kwargs={
                        "llm_runable": self.llm_runable if self.kor_dreams_task_step_llm is None else self.kor_dreams_task_step_llm,
                        "system_prompt": gpt_prompt_config.evaluate_system_prompt,
                        "user_prompt": user_prompt,
                        "storage_context": self.storage_context,
                    },
                )

                _ai_message = results[0]

                owner = (
                    f"event end thread {threading.get_native_id()}, _evaluate_answer"
                )
                logger.info(f"owner:{owner}")
                assert _ai_message.content is not None
                cleaned_text = re.sub(r'◁think▷.*?◁/think▷', '',_ai_message.content, flags=re.DOTALL)
                return int(cleaned_text)
            except ValueError:
                user_prompt = (
                    f"{_ai_message.content}\n\nFailed to parse reward as an integer."
                )

                if attempt == 2:
                    raise

    def _kor_task_step_refine_builder(
        self, refined_answer_response_message: AIMessage
    ) -> list[TaskStepRefineNode]:
        """
        抽取根据批评意见优化当前回答并续写上下文内容
        """
        kor_task_step_refine_builder = KorLoader.form_kor_task_step_refine_builder(
            llm_runable=self.llm_runable if self.kor_dreams_task_step_llm is None else self.kor_dreams_task_step_llm,
        )
        response = kor_task_step_refine_builder.run(
            refined_answer_response_message.content
        )
        task_step_refine_node_list = []
        if (
            response.get("data") is not None
            and response.get("data").get("script") is not None
        ):
            step_list = response.get("data").get("script")
            for step in step_list:
                task_step_refine_node = TaskStepRefineNode(
                    thought=step.get("thought"),
                    answer=step.get("answer"),
                    answer_socre=step.get("answer_socre"),
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


def _build_structured_storyboard(
    task_step_store: BaseTaskStepStore,
) -> StructuredStoryboard:
    task_step_all = task_step_store.task_step_all
    task_step_all_list = [val.__dict__ for val in list(task_step_all.values())]
    structured_storyboard = StructuredStoryboard(json_data=task_step_all_list)

    return structured_storyboard


# 链表转树的函数
def linked_list_to_tree(linked_list_head):
    if linked_list_head is None:
        return None

    # 递归构建树
    def build_tree(node: LinkedListNode, parent: MCTSNode = None):
        if node is None:
            return None

        # 为当前链表节点创建树节点
        mcts_node = MCTSNode(
            task_step_id=node.task_step_id,
            answer=node.task_step_question_answer,
            children=[],
            visits=0,
            Q=0,
            reward_samples=[],
            parent=parent,
        )

        # 如果有下一个节点，继续递归构建子节点
        if node.next:
            child_node = build_tree(node.next, mcts_node)
            mcts_node.children.append(child_node)

        return mcts_node

    # 从链表头开始构建树
    root = build_tree(linked_list_head)
    return root


# 使用栈来模拟深度优先搜索
def find_matching_node_in_tree_iterative(tree_node, linked_list_node):
    # 初始化栈，开始时栈中只包含根节点
    stack = [tree_node]

    while stack:
        # 获取栈顶的节点
        current_node = stack.pop()

        # 如果当前节点与链表节点相同，返回当前节点
        if current_node.answer == linked_list_node.task_step_question_answer:
            return current_node

        # 将子节点添加到栈中，按深度优先顺序处理
        for child in current_node.children:
            stack.append(child)

    return None  # 如果没有匹配的节点，返回 None
