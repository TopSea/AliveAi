import datetime
import json
import os
import sys
import threading
import time
import requests
from alive.api_llm import AliveMessage, OllamaMessage, ollama_gen
from alive.api_tts import cosy_tts_gen
from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.playback import play
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dataclasses import dataclass
from typing import List, Optional
from RealtimeSTT import AudioToTextRecorder
import uvicorn

from alive.alive_config import AliveConfig, initialize_config_file


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


tts_queue = []


def tts_gen_queue():
    print("tts_queue start")
    while True:
        if len(tts_queue) > 0:
            print("tts_queue running")
            txt, index = tts_queue.pop(0)
            cosy_tts_gen(txt, "alive_temp", index)

        time.sleep(1)


# thread = threading.Thread(target=tts_gen_queue, args=())
# thread.start()


# 定义请求体的 Pydantic 模型
class AliveChat(BaseModel):
    model: str
    prompt: str
    stream: bool


# 定义 POST 请求的 endpoint
@app.post("/alive_chat/")
async def forward_to_ollama(request: AliveChatRequest):
    # Ollama API 的 URL
    ollama_api_url = alive_config.get("llm")["ollama"]["ollama_api"]
    txt = ""
    index = 0
    print(request)
    # 发送请求
    try:
        # 将请求转发到 Ollama API，并获取流式响应
        llm_response = requests.post(
            ollama_api_url,
            json=request.model_dump(),  # 将 Pydantic 模型转换为字典
        )

        # 检查是否出现 HTTP 错误
        llm_response.raise_for_status()

        # 播放音频
        for line in llm_response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                data: OllamaMessage = json.loads(decoded_line)
                # print(data)
                if "message" in data:
                    txt += data["message"]["content"]
                    if txt.count("。") >= 2:
                        print(txt)
                        tts_queue.append((txt, index))
                        index += 1
                        txt = ""

        # 定义一个生成器，用于逐块读取流式响应
        async def stream_generator():
            for chunk in llm_response.iter_content(chunk_size=None):
                if chunk:  # 过滤掉保持活跃的空 chunk
                    yield chunk.decode("utf-8")  # 假设响应是 UTF-8 编码

        # 返回流式响应
        return StreamingResponse(stream_generator(), media_type="text/plain")

    except requests.HTTPError as e:
        # 处理 HTTP 请求错误
        raise HTTPException(
            status_code=500, detail=f"Error forwarding request to Ollama API: {str(e)}"
        )
    except requests.RequestException as e:
        # 处理其他请求异常
        raise HTTPException(
            status_code=500, detail=f"Request to Ollama API failed: {str(e)}"
        )


# 定义 POST 请求的 endpoint
@app.post("/update_config/")
async def forward_to_ollama(config_str: str = Form()):
    config = json.loads(config_str)
    print(config)
    alive_config.update(config)
    pass


def process_text(text):
    succeed = False
    # 执行命令
    # if text != None and len(text) > 0:
    #     matched = create_command(text)
    #     succeed = match_and_exec(matched)

    def push_tts(txt, index):
        tts_queue.append((txt, index))

    # 执行成功就不再对话
    if not succeed:
        print(text)
        ollama_gen(text, push_tts)


if __name__ == "__main__":
    # 初始化配置文件
    # try:
    #     initialize_config_file("alive_ai_config.json")
    #     alive_config = AliveConfig()
    # except Exception as e:
    #     print(f"配置文件初始化失败: {e}")
    #     sys.exit(1)  # 退出程序

    # 启动 tts_gen_queue 线程
    tts_thread = threading.Thread(target=tts_gen_queue, args=())
    tts_thread.daemon = True  # 设置为守护线程，主程序退出时自动终止
    tts_thread.start()

    # 启动 uvicorn 服务器线程
    def run_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    uvicorn_thread = threading.Thread(target=run_uvicorn)
    uvicorn_thread.daemon = True  # 设置为守护线程
    uvicorn_thread.start()

    with AudioToTextRecorder(
        wakeword_backend="oww",
        wake_words_sensitivity=0.35,
        openwakeword_model_paths="./model/wakeword/furina.onnx",
        wake_word_buffer_duration=0.5,
        model="medium",
        language="zh",
        device="cuda",
    ) as recorder:

        print(f'Say "Furina" to start recording.')
        while True:
            recorder.text(process_text)
