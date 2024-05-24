# InterpretationoDreams

## Description
使用langchain实现 故事情景生成，情感情景引导，剧情总结，性格分析


### 数据处理方式
相关工具
- `https://github.com/glide-the/Keyframe-Extraction-for-video-summarization` 提取视频关键帧,结合字幕文件整理分镜信息
- `https://github.com/LC1332/Chat-Haruhi-Suzumiya` 提取视频字幕及其对应的时间戳

 
主要模块
- 个体心理状态提取思维链： 通过分析剧本和结果，实现了结构化的个体内在心理状态及其行为的驱动力。情感细节从原本的字幕中被有效提取，加载对话加载器以进行角色扮演。
- 音视频处理模块： 调用了 TransNetV2 和 whispe 音视频处理模型，封装了完整的数据加载代码，以满足后续思维链加载及数据分析的需求。
- 设计实现会话加载器： 通过对 langchain 的每个代码块进行抽象，构建了一个代码生成器，将langchai 工具分成基础程序、逻辑控制程序、逻辑加载程序、逻辑运行程序。这个设计使得在应用数据模块概念后更加实用和简单。
- 设计记忆模块： 自定义了不同形态的记忆存储形式，为后续行动决策与环境感知的任务协同分析提供了关键指标，同时增强了人类对行为的可解释性和可信度。


##### 数据整理 [README.md](src/docs/README.md)
