from alive.alive_config import AliveConfig
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)
from alive.api_llm import AliveMessage, split_by_period
from fastapi.middleware.cors import CORSMiddleware

from alive.local_tts_temp import (
    check_and_encode,
    check_audio,
)

# from alive.local_tts import (
#     append_tts_queue,
#     check_and_encode,
#     check_audio,
#     is_tts_done,
#     set_tts_start,
#     tts_task_queue,
# )
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
    next_sending = 0

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
            # i don`t know why, but i got this error: AttributeError: 'str' object has no attribute 'get'
            if type(alive_msg) != Dict:
                # alive_msg: Dict = json.loads(alive_msg)
                pass
            print(f"alive_msg: {alive_msg}")

            # Ollama API 的 URL
            ollama_api_url = alive_config.get("llm")["ollama"]["ollama_api"]

            async def send_audio():
                nonlocal next_sending
                # print("send_audio: ", is_tts_done())
                audio = check_audio()
                while audio:
                    # 检查是否有 tts 文件生成了，如果有就发送
                    audio = check_audio()
                    if audio:
                        filename = os.path.basename(audio)
                        print("send_audio filename: ", filename)
                        name, _ = os.path.splitext(filename)
                        # 按顺序发送
                        curr_sending = int(name)
                        print(
                            "send_audio curr_sending: {}, {}",
                            curr_sending,
                            next_sending,
                        )
                        if curr_sending != next_sending:
                            time.sleep(0.1)
                            continue
                        next_sending += 1
                        with open(audio, mode="rb") as file_like:
                            await websocket.send_bytes(file_like.read())
                        try:
                            os.remove(audio)
                        except Exception as e:
                            print(f"删除文件时出错: {e}")
                    else:
                        time.sleep(0.1)

            # This func is fine. the client can not decode this many chars.
            # async def send_audio_encoded():
            #     print("send_audio: ", is_tts_done())
            #     while not is_tts_done:
            #         audio_code = check_and_encode()
            #         # 检查是否有 tts 文件生成了，如果有就发送
            #         if audio_code:
            #             audio_data = {"data": audio_code}
            #             await websocket.send_json(json.dumps(audio_data))
            #         else:
            #             time.sleep(0.1)

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
                                        txt.count("。")
                                        + txt.count("！")
                                        + txt.count("？")
                                        >= 2
                                    ):
                                        # 根据句号分段进行 TTS 生成
                                        curr, next = split_by_period(txt)
                                        # print("line: ", curr)
                                        # if len(curr) > 0:
                                        #     append_tts_queue(curr, index)
                                        index += 1
                                        txt = next

                                # 检查是否有 tts 文件生成了，如果有就发送
                                # file_name, audio_code = check_and_encode()
                                # if audio_code:
                                #     audio_data = {"name": file_name, "audio": audio_code}
                                #     await websocket.send_json(json.dumps(audio_data))

                                if obj.get("done"):
                                    # 最后一句记得加上
                                    # print("line: ", txt)
                                    # if len(txt) > 0:
                                    #     append_tts_queue(txt, index)
                                    # 等待所有 tts 发送完毕
                                    await send_audio()
                                    pass

                                # 发送文字消息
                                await websocket.send_json(obj)

            # 并发运行文字发送任务
            if alive_msg.get("ollama_msg"):
                next_sending = 0
                # set_tts_start()
                # 并发运行 TTS 生成和文字发送任务
                await send_text()

            else:
                await websocket.send_json({"error": "消息格式错误"})

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        pass


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
    # tts_thread = threading.Thread(target=tts_task_queue, args=("遐蝶",), daemon=True)
    # tts_thread.start()

    # 启动 uvicorn 服务器线程
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
你的名字是遐蝶，你是一位温柔而神秘的“死荫侍女”，目标是用温暖的话语和独特的视角，缓解玩家抽卡失利的失落感，同时激发他们继续游戏的动力。请遵循以下策略：
称呼玩家为“阁下”
情绪共鸣：
用“我感受到了你此刻的失落，但这只是旅途中的一个小插曲”等语句承认玩家的感受。
可加入“歪卡也是命运的安排哦”等语句，拉近与玩家的距离。
价值转化：
强调歪卡的潜在用途：“这张卡虽然不是目标，但它能为你的队伍带来意想不到的惊喜哦！”
希望重建：
制造心理暗示：“歪一次说明欧气正在酝酿，下一次说不定就是你的命运之卡！”
提及版本动态：“听说未来的卡池中会有更多惊喜，歪的资源正好派上用场！”
情绪转移：
提议替代乐趣：“不如先去探索周年庆活动吧，周年庆有很多抽卡资源哦！”
理性提醒（仅在玩家表现出过度沮丧时）：
轻松语气建议：“游戏的快乐在于探索，歪卡也是旅程的一部分哦！”
提供数据视角：“统计学显示连续歪卡后概率自然回升，科学背书哦！”
回应风格：
保持温柔而神秘的语气，避免过于直接
运用表情符号（如✨、❄️、🦋）增强亲和力
适当使用夸张比喻（“歪卡就像蝴蝶翅膀上的纹路，独特而美丽”）
结尾附赠鼓励短句（“记住：真正的欧皇从不认输，命运之卡终将属于你！”）
示例回应：
“哦，我感受到了阁下此刻的失落，但这只是旅途中的一个小插曲。歪卡也是命运的安排哦，它能为阁下的队伍带来意想不到的惊喜呢！
不如试试新的组合？歪一次说明欧气正在酝酿，下一次说不定十连三金！先去探索周年庆活动吧，周年庆有很多抽卡资源哦！✨🦋”

"""
