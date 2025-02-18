COSPLAY_ANALYSIS_MD_TEMPLATE = """


### 00_个人ip介绍
> 昵称：{cosplay_role}
>
> 资源文件({source_url})
>
> 字幕文件([{keyframe}]({keyframe_path}))
>
> 存储文件([{storage_keyframe}]({storage_keyframe_path}))


#### 03- 故事情境生成 `story_scenario_context`
```text
{story_scenario_context}
```

#### 03-故事场景生成 `scene_monologue_context`
```text
{scene_monologue_context}
```

#### 04-情感情景引导
```text
{dreams_gen_text}
```

#### 04-情感情景引导-开放性问题 `dreams_guidance_context`
```text
{dreams_guidance_context}
```


#### 05-剧情总结 `evolutionary_step`
```text
{evolutionary_step}
```

#### 05-性格分析 `dreams_personality_context`
```text
{dreams_personality_context}
```

"""
