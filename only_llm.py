from dotenv import load_dotenv
from alive.alive_config import AliveConfig
from fastapi import (
    FastAPI,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from alive.alive_nlp import emotion_classify
from alive.alive_util import decode_config_from_alive
from alive.api_llm import AliveMessage
from fastapi.middleware.cors import CORSMiddleware
from alive.local_tts_temp import (
    check_audio,
)

from typing import Dict, List
from pydantic import BaseModel
import aiohttp
import uvicorn
import json
import time
import os

# 加载 .env 文件
load_dotenv()
# 读取 LOCK_CONFIG_PWD 的值
lock_config_pwd = os.environ.get("LOCK_CONFIG_PWD")


class AliveChatRequest(BaseModel):
    model: str
    messages: List[AliveMessage]


# 获取全局配置实例
alive_config: AliveConfig


# 创建 FastAPI 应用
app = FastAPI()
# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义请求体的 Pydantic 模型
class AliveChat(BaseModel):
    model: str
    prompt: str
    stream: bool


# 接收来自 Alive 的更新设置请求
@app.post("/update_config/")
async def forward_to_ollama(config_str: str = Form()):
    decoded_str = decode_config_from_alive(config_str)
    config = json.loads(decoded_str)
    print(config)
    if len(lock_config_pwd) > 0:
        if lock_config_pwd == config.get("config_pwd"):
            alive_config.update(config)
    else:
        alive_config.update(config)


@app.websocket("/alive_talk/{client_id}")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 接收客户端消息
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "text" in message:
                    text = message["text"]
                    # print(f"Received text: {text}")
                elif "bytes" in message:
                    bytes_data = message["bytes"]
                    print(f"Received bytes: {bytes_data}")
                else:
                    print("Unknown message type")
            elif message["type"] == "websocket.disconnect":
                print("Client disconnected")
                break

            if text == "ping":
                time.sleep(0.1)
                continue
            alive_msg = json.loads(text)
            print(f"alive_msg: {alive_msg}")

            # 并发运行文字发送任务
            if alive_msg.get("ollama_msg"):
                # 并发运行 TTS 生成和文字发送任务
                await send_text(alive_msg, websocket)

            else:
                await websocket.send_json({"error": "消息格式错误"})

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        pass


# 主线程处理文字消息并发送任务到伴生线程
async def send_text(alive_msg: dict, websocket: WebSocket):
    ollama_api_url = alive_config.get("llm")["ollama"]["ollama_api"] + "/api/chat/"
    txt: str = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            ollama_api_url,
            json=alive_msg.get("ollama_msg"),
        ) as response:
            response.raise_for_status()

            async for line in response.content.iter_any():
                decoded_line = line.decode("utf-8")
                # 有时候会阻塞，然后传来一堆 json
                decoder = json.JSONDecoder()
                pos = 0
                while pos < len(decoded_line):
                    print(f"decoded_line: {decoded_line}, pos: {pos}")
                    obj, pos = decoder.raw_decode(decoded_line, pos)
                    pos += 1

                    if "message" in obj:
                        txt += obj.get("message")["content"]

                    if obj.get("done"):
                        emotion = emotion_classify(txt)
                        print(f"emotion: {emotion}")
                        await websocket.send_text(emotion)

                    # 发送文字消息
                    await websocket.send_json(obj)


if __name__ == "__main__":
    alive_config = AliveConfig()
    # 初始化配置文件
    # try:
    #     initialize_config_file("alive_ai_config.json")
    #     alive_config = AliveConfig()
    # except Exception as e:
    #     print(f"配置文件初始化失败: {e}")
    #     sys.exit(1)  # 退出程序

    # 启动 uvicorn 服务器线程
    uvicorn.run(app, host="0.0.0.0", port=20167)
