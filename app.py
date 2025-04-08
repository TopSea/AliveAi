from alive.alive_config import AliveConfig
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from alive.api_llm import AliveMessage, split_by_period
from fastapi.middleware.cors import CORSMiddleware
from alive.local_tts import (
    append_tts_queue,
    check_audio,
    is_tts_done,
    set_tts_start,
    tts_task_queue,
)
from cosyvoice.cli.cosyvoice import CosyVoice2
from typing import Dict, List
from pydantic import BaseModel
import threading
import aiohttp
import uvicorn
import json
import time

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


@app.websocket("/alive_talk")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # 接收客户端消息
        msg = await websocket.receive_text()
        alive_msg: Dict = json.loads(msg)

        print(f"Received message: {alive_msg}")
        # Ollama API 的 URL
        ollama_api_url = alive_config.get("llm")["ollama"]["ollama_api"]

        async def send_audio():
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

        # 主线程处理文字消息并发送任务到伴生线程
        async def send_text():
            txt: str = ""
            index = 0
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
                                if (
                                    txt.count("。") + txt.count("！") + txt.count("？")
                                    >= 2
                                ):
                                    # 根据句号分段进行 TTS 生成
                                    curr, next = split_by_period(txt)
                                    print("line: ", curr)
                                    if len(curr) > 0:
                                        append_tts_queue(curr, index)
                                    index += 1
                                    txt = next

                            # 检查是否有 tts 文件生成了，如果有就发送
                            audio = check_audio()
                            if audio:
                                with open(audio, mode="rb") as file_like:
                                    await websocket.send_bytes(file_like.read())
                                try:
                                    os.remove(audio)
                                except Exception as e:
                                    print(f"删除文件时出错: {e}")

                            if obj.get("done"):
                                # 最后一句记得加上
                                print("line: ", txt)
                                if len(txt) > 0:
                                    append_tts_queue(txt, index)
                                # 等待所有 tts 发送完毕
                                await send_audio()
                                pass

                            # 发送文字消息
                            await websocket.send_json(obj)

        # 并发运行文字发送任务
        if alive_msg.get("ollama_msg"):
            set_tts_start()
            # 并发运行 TTS 生成和文字发送任务
            await send_text()

        else:
            await websocket.send_json({"error": "消息格式错误"})

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        pass


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

    # 启动 tts_gen_queue 线程
    tts_thread = threading.Thread(target=tts_task_queue, args=())
    tts_thread.daemon = True  # 设置为守护线程，主程序退出时自动终止
    tts_thread.start()

    # 启动 uvicorn 服务器线程
    uvicorn.run(app, host="0.0.0.0", port=8000)
