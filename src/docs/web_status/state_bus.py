"""
事件总线：供主项目零侵入接入
"""
import asyncio, json
from aiohttp import web
from weakref import WeakSet

# 全局 SSE 响应集合
_listeners: WeakSet[web.StreamResponse] = WeakSet()

async def _broadcast(state: str):
    dead = set()
    msg = f"data: {json.dumps({'state': state})}\n\n"
    for resp in _listeners:
        try:
            await resp.write(msg.encode('utf-8'))
        except Exception:
            dead.add(resp)
    for d in dead:
        _listeners.discard(d)

def emit_state(state: str):
    """同步/异步两用"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_broadcast(state))
        else:
            loop.run_until_complete(_broadcast(state))
    except RuntimeError:
        asyncio.run(_broadcast(state))

async def register(resp):
    _listeners.add(resp) 