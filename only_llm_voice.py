from dotenv import load_dotenv
from faster_whisper import WhisperModel
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

from typing import List
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


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_size = "medium"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Whisper 热身
_, _ = model.transcribe("./asset/芙宁娜.mp3", beam_size=5)


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


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_personal_json(self, json_dict: dict, websocket: WebSocket):
        await websocket.send_json(json_dict)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/alive_talk/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            voice_file = f"./asset/{client_id}.webm"
            text = ""
            voice_file_write = False

            # receive clients messages
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "text" in message:
                    text = message["text"]
                    print(f"Received text: {text}")
                elif "bytes" in message:
                    voice_data = message["bytes"]
                    with open(voice_file, "wb") as f:
                        f.write(voice_data)
                    voice_file_write = True
                else:
                    print("Unknown message type")
            elif message["type"] == "websocket.disconnect":
                print("Client disconnected")
                break

            if text == "ping":
                time.sleep(0.1)
                continue
            if text != "":
                # history messages
                alive_msg: dict = json.loads(text)
                history: list = alive_msg["ollama_msg"]["messages"]

            if voice_file_write:
                segments, _ = model.transcribe(voice_file, beam_size=5, vad_filter=True)
                voice_text = ""
                for segment in segments:
                    voice_text += segment.text
                voice_text = voice_text.strip()
                os.remove(voice_file)

                if voice_text == "":
                    # no message in voice data
                    continue
                print(f"voice_text: {voice_text}")

                new_message = {
                    "role": "user",
                    "content": voice_text,
                }
                history.append(new_message)

                user_message = {
                    "message": new_message,
                    "done": True,
                }
                await manager.send_personal_message(json.dumps(user_message), websocket)

            if len(history) > 0 and history[-1]["role"] == "user":
                await send_text(alive_msg, websocket)
            else:
                continue

            # combine voice message and history message, and then send to ollama

            # await manager.send_personal_message(f"You wrote: {data}", websocket)
            # await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client #{client_id} left the chat")


async def send_text(alive_msgs: dict, websocket: WebSocket):
    txt: str = ""
    print(f"alive_msgs: {alive_msgs}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            ollama_api_url,
            json=alive_msgs.get("ollama_msg"),
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
                        pass

                    # 发送文字消息
                    await websocket.send_json(obj)


if __name__ == "__main__":
    alive_config = AliveConfig()
    ollama_api_url = alive_config.get("llm")["ollama"]["ollama_api"] + "/api/chat/"

    # 启动 uvicorn 服务器线程
    uvicorn.run(app, host="0.0.0.0", port=20167)
