from alive.alive_config import AliveConfig, initialize_config_file
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from alive.api_llm import AliveMessage, OllamaMessage, ollama_gen
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from cosyvoice.cli.cosyvoice import CosyVoice2
from alive.api_tts import cosy_tts_gen
from dataclasses import dataclass
from typing import Dict, List, Optional
from pydub.playback import play
from pydantic import BaseModel
from pydub import AudioSegment
from asyncio import Lock
from queue import Queue
import numpy as np
import threading
import datetime
import requests
import asyncio
import aiohttp
import uvicorn
import torch
import json
import time
import sys
import io
import os


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


tts_queue = Queue()
tts_result_queue = Queue()


def audio_generation_thread(websocket: WebSocket):
    prompt_speech_16k = load_voice_data(alive_config.get("tts")["cosy"]["speaker"])
    if prompt_speech_16k is None:
        websocket.send_json({"error": "预训练音色文件中缺少audio_ref数据！"})
        # tts_result_queue.put({"error": "预训练音色文件中缺少audio_ref数据！"})
    while True:
        print("tts_queue start")
        task = tts_queue.get()
        # if task is None:  # 退出信号
        #     print(f"audio_generation_thread done")
        #     # tts_result_queue.put({"done": True})
        #     websocket.send_json({"done": True})
        #     break
        txt = task.get("txt")
        print(f"audio_generation_thread: Processing text: {txt}")
        if not txt:  # 如果没有内容，直接返回
            time.sleep(0.1)
            continue

        try:
            # 音频生成逻辑
            model_output = cosyvoice.inference_instruct2(
                txt,
                alive_config.get("tts")["cosy"]["instruct_text"],
                prompt_speech_16k,
                speed=alive_config.get("tts")["cosy"]["speed"],
            )
            audio_data = generate_data(model_output)
            websocket.send_bytes(audio_data)
            # tts_result_queue.put({"data": audio_data})
        except Exception as e:
            print(f"Error generating audio: {e}")
            # tts_result_queue.put({"error": f"音频生成失败: {str(e)}"})
            websocket.send_json({"error": f"音频生成失败: {str(e)}"})
            break

        time.sleep(0.1)


def load_voice_data(speaker):
    """加载语音数据"""
    voice_path = f"./models/tts_voices/{speaker}.pt"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(voice_path):
            return None
        voice_data = torch.load(voice_path, map_location=device)
        return voice_data.get("audio_ref")
    except Exception as e:
        raise ValueError(f"加载音色文件失败: {e}")


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
        yield tts_audio


# 启动伴生线程
def start_audio_thread(websocket: WebSocket):
    thread = threading.Thread(target=audio_generation_thread, args=[websocket])
    thread.daemon = True  # 设置为守护线程，随主线程退出
    thread.start()
    return thread


@app.websocket("/alive_talk")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 启动伴生线程
    audio_thread = start_audio_thread(websocket)

    try:
        # 接收客户端消息
        msg = await websocket.receive_text()
        alive_msg: Dict = json.loads(msg)

        print(f"Received message: {alive_msg}")
        # Ollama API 的 URL
        ollama_api_url = alive_config.get("llm")["ollama"]["ollama_api"]

        # 主线程处理文字消息并发送任务到伴生线程
        async def send_text():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ollama_api_url,
                    json=alive_msg.get("ollama_msg"),
                ) as response:
                    response.raise_for_status()

                    async for line in response.content.iter_any():
                        data: Dict = json.loads(line)
                        print(data)

                        if "message" in data:
                            txt = data.get("message")["content"]
                            # 将音频生成任务放入队列
                            tts_queue.put({"txt": txt})
                        if data.get("done"):
                            pass

                        # 发送文字消息
                        await websocket.send_json(data)

        async def gen_tts():
            while True:
                print("gen_tts")
                result = tts_queue.get()
                if result:
                    if "data" in result:
                        await websocket.send_bytes(result["data"])
                    elif "done" in result:
                        break
                    elif "error" in result:
                        await websocket.send_json({"error": result["error"]})
                else:
                    time.sleep(0.1)
                    continue
            return "done"

        # 并发运行文字发送任务
        if alive_msg.get("ollama_msg"):
            # await asyncio.gather(send_text(), gen_tts())
            await send_text()

        else:
            await websocket.send_json({"error": "消息格式错误"})

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        # 停止伴生线程
        tts_queue.put(None)
        audio_thread.join(timeout=1)


if __name__ == "__main__":
    cosyvoice = CosyVoice2("models/CosyVoice2-0.5B")
    alive_config = AliveConfig()
    # 初始化配置文件
    # try:
    #     initialize_config_file("alive_ai_config.json")
    #     alive_config = AliveConfig()
    # except Exception as e:
    #     print(f"配置文件初始化失败: {e}")
    #     sys.exit(1)  # 退出程序

    # 启动 uvicorn 服务器线程
    uvicorn.run(app, host="0.0.0.0", port=8000)
