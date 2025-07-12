#  f30c 实时状态监控页面

此目录提供一个 **零依赖浏览器端** 的实时状态图 Demo，用于可视化 InterpretationoDreams 的运行流程。

## 1. 快速启动（本地）
```bash
# 1) 进入目录
cd web_status/

# 2) 安装依赖
pip install -r requirements.txt

# 3) 启动 SSE 服务器
python status_server.py
```

浏览器访问 http://localhost:8080  
你会看到节点依次高亮，代表任务逐步推进。

## 2. 与主项目集成
在任意 Python 模块里 **两行代码** 即可把真实状态推送到前端：

```python
from web_status.state_bus import emit_state

emit_state("TaskPlanning")   # 进入状态
# ... 处理逻辑 ...
emit_state("TaskStepGen")    # 离开状态
```

> 如果主项目已使用 `asyncio`，`emit_state` 会自动检测事件循环并异步发送；否则退化为同步调用。

## 3. 自定义状态图
- 修改 `state.dot`（Graphviz DOT 语法）→ 刷新页面立即生效。  
- 节点颜色、动画时长可在 `index.html` 的 `<style>` 里调整。

## 4. 生产部署
```bash
# 使用 uvicorn + gunicorn
gunicorn status_server:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:80
```

## 5. 开发提示
- 浏览器控制台会打印每一次 SSE 数据，方便调试。  
- 状态名必须与 `state.dot` 里的 `id` 完全一致（区分大小写）。 