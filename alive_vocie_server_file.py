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
from alive.api_llm import AliveMessage, split_by_period
from fastapi.middleware.cors import CORSMiddleware

# from alive.local_tts_temp import (
#     check_and_encode,
#     check_audio,
# )

from alive.local_tts import (
    append_tts_queue,
    tts_task_queue_zero,
    check_and_encode,
    check_audio,
    is_tts_done,
    set_tts_start,
    tts_task_queue,
)
from typing import Dict, List
from pydantic import BaseModel
import threading
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

whisper_model = "./pretrained_models/faster-whisper-small"
# Run on GPU with FP16
model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
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


@app.websocket("/alive_talk/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()

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
                await websocket.send_text(json.dumps(user_message))

            if len(history) > 0 and history[-1]["role"] == "user":
                set_tts_start()
                await send_text(alive_msg, websocket)
            else:
                continue

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        pass


async def send_text(alive_msg: dict, websocket: WebSocket):
    txt: str = ""
    totalTxt: str = ""
    index = 0
    ollama_api_url = alive_config.get("llm")["ollama"]["ollama_api"] + "/api/chat"
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
                        totalTxt += obj.get("message")["content"]
                        if len(txt) >= 30:
                            append_tts_queue(txt, index)
                            index += 1
                            txt = ""

                    if obj.get("done"):
                        emotion = emotion_classify(txt)
                        print(f"emotion: {emotion}")
                        await websocket.send_text(emotion)
                        # 最后一句记得加上
                        print("line: ", txt)
                        if len(txt) > 0:
                            append_tts_queue(txt, index)
                        # 等待所有 tts 发送完毕
                        await send_audio(websocket)
                        pass

                    # 发送文字消息
                    await websocket.send_json(obj)


async def send_audio(websocket: WebSocket):
    print("send_audio: ", is_tts_done())
    # audio = check_audio()
    while not is_tts_done():
        # 检查是否有 tts 文件生成了，如果有就发送
        audio = check_audio()
        if audio:
            with open(audio, mode="rb") as file_like:
                await websocket.send_bytes(file_like.read())
            try:
                os.remove(audio)
            except Exception as e:
                print(f"删除文件时出错: {e}")
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    alive_config = AliveConfig()
    # 初始化配置文件
    # try:
    #     initialize_config_file("alive_ai_config.json")
    #     alive_config = AliveConfig()
    # except Exception as e:
    #     print(f"配置文件初始化失败: {e}")
    #     sys.exit(1)  # 退出程序

    # 启动 tts_gen_queue 线程
    tts_thread = threading.Thread(
        target=tts_task_queue_zero, args=("薇薇安",), daemon=True
    )
    tts_thread.start()

    # 启动 uvicorn 服务器线程
    uvicorn.run(app, host="0.0.0.0", port=20167)
