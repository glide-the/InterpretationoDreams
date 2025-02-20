"""

"""

# 子任务指令转换为子问题.txt
CONVERT_TASK_STEP_TO_QUESTION_PROMPT_TEMPLATE = """执行任务：结合开始任务（start_task_context），在符合任务总体描述（aemo_representation_context）的情况下，根据任务步骤名称（task_step_name）、任务步骤描述（task_step_description）和任务步骤层级（task_step_level），提出一个问题,不要解释直接输出问题,使用英文输出
# 结果使用英文回复

### 任务

start_task_context: {start_task_context}

aemo_representation_context: {aemo_representation_context}


### 当前任务信息

task_step_name: {task_step_name}
task_step_description: {task_step_description}
task_step_level: {task_step_level}
"""


TASK_STEP_QUESTION_TO_GRAPHQL_PROMPT_TEMPLATE = """改写问题为graphql语句，主要任务如下
1 、分析问题中概念，找出概念的语义词，
如 我想吃有煎蛋、火腿片和烤豆

concepts["food"] 

2、根据概念语义词(concepts) 分析，列出远离这个概念的任意一个词，靠近这个概念的若干词
如 我想吃有煎蛋、火腿片和烤豆

concepts["food"] 

moveAwayFrom: {{
  concepts: ["finance"], 
  force: 0.45
}}

moveTo: {{
  concepts: ["egg", "food"], 
  force: 0.85 
}}


3、 将概念和远离靠近的概念，生成"graphql语句" 任务

### 环境信息

RootQuery: "Get",
class: "{collection_name_context}", 
operators: "nearText" 

### Data Schema
```
(
    "{collection_name_context}",
     
    properties=[  # Define properties
        Property(name="refId", data_type=DataType.TEXT),
        Property(name="chunkText", data_type=DataType.TEXT), 
        
    ],
)
```

#### nearText Variables 

| Variable                 | Required | Type       | Description                                                  |
| ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
| `concepts`               | yes      | `[string]` | An array of strings that can be natural language queries, or single words. If multiple strings are used, a centroid is calculated and used. Learn more about how the concepts are parsedhere. |
| `distance`               | no       | `float`    | The maximum allowed distance to the provided search input. Cannot be used together with the `certainty` variable. The interpretation of the value of the distance field depends on the distance metric used. |
| `certainty`              | no       | `float`    | Normalized Distance between the result item and the search vector. Normalized to be between 0 (perfect opposite) and 1 (identical vectors). Can't be used together with the `distance` variable. |
| `autocorrect`            | no       | `boolean`  | Autocorrect input text values. Requires the `text-spellcheck` module to be present & enabled. |
| `moveTo`                 | no       | `object<>` | Move your search term closer to another vector described by keywords |
| `moveTo<concepts>`       | no       | `[string]` | An array of strings - natural language queries or single words. If multiple strings are used, a centroid is calculated and used. |
| `moveTo<objects>`        | no       | `[UUID]`   | Object IDs to move the results to. This is used to "bias" NLP search results into a certain direction in vector space. |
| `moveTo<force>`          | no       | `float`    | The force to apply to a particular movement. Must be between 0 and 1 where 0 is equivalent to no movement and 1 is equivalent to largest movement possible. |
| `moveAwayFrom`           | no       | `object<>` | Move your search term away from another vector described by keywords |
| `moveAwayFrom<concepts>` | no       | `[string]` | An array of strings - natural language queries or single words. If multiple strings are used, a centroid is calculated and used. |
| `moveAwayFrom<objects>`  | no       | `[UUID]`   | Object IDs to move the results from. This is used to "bias" NLP search results into a certain direction in vector space. |
| `moveAwayFrom<force>`    | no       | `float`    | The force to apply to a particular movement. Must be between 0 and 1 where 0 is equivalent to no movement and 1 is equivalent to largest movement possible. |


### Semantic Path

Example: showing a semantic path without edges.
```

_additional {{
  semanticPath {{
    path {{
      concept
      distanceToNext
      distanceToPrevious
      distanceToQuery
      distanceToResult
    }}
  }}
}}

```


### 注意
请不要修改环境信息


### 结果示例
```
{{
  Get {{
    {collection_name_context}(
      nearText:{{
        concepts: ["food"], 
        distance: 0.23, 
        moveAwayFrom: {{
          concepts: ["finance"],
          force: 0.45
        }},
        moveTo: {{
          concepts: ["egg", "food"],
          force: 0.85
        }}
      }}, 
      limit: 25
    ) {{
     
     refId
     chunkText
     _additional {{
        semanticPath {{
          path {{
            concept
            distanceToNext
            distanceToPrevious
            distanceToQuery
            distanceToResult
          }}
        }}
      }}
    }}
  }}
}}

```



下面执行这个问题相关的任务
{task_step_question}
"""
