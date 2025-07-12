#!/usr/bin/env python3
"""
运行：python status_server.py
"""
import asyncio, json, random
from pathlib import Path
from aiohttp import web

# 状态机顺序（和 PlantUML 保持一致）
STATES = [
    "InitEnv", "LoadConfig", "InitLLMClient", "Idle",
    "TaskPlanning", "TaskStepGen", "TaskStepStore", "TaskReady",
    "SceneBuild", "AemoPrompt", "EdreamsPrompt", "ResourcePool",
    "MCTSLoop", "AnswerMerge", "PostRank", "PresentResult",
    "UserFeedback", "UpdateReward"
]

routes = web.RouteTableDef()

@routes.get('/events')
async def events(request):
    resp = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )
    await resp.prepare(request)
    for st in STATES:
        await asyncio.sleep(random.uniform(0.4, 1.2))  # 模拟真实耗时
        msg = f"data: {json.dumps({'state': st})}\n\n"
        await resp.write(msg.encode('utf-8'))
    # 回到空闲
    msg = f"data: {json.dumps({'state': 'Idle'})}\n\n"
    await resp.write(msg.encode('utf-8'))
    await resp.write_eof()
    return resp

@routes.get('/')
async def index(request):
    return web.FileResponse(Path(__file__).with_name("index.html"))

@routes.get('/state.dot')
async def dot(request):
    return web.FileResponse(Path(__file__).with_name("state.dot"))

app = web.Application()
app.router.add_routes(routes)
web.run_app(app, host="0.0.0.0", port=8080) 