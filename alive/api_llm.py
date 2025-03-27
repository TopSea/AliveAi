import json
import requests
from pydantic import BaseModel
from typing import List, Optional
from alive.alive_config import AliveConfig


class AliveMessage(BaseModel):
    role: str
    content: str
    # images: Optional[list] = None  # images 可以是 None 或者一个列表


class OllamaMessage(BaseModel):
    model: str
    created_at: str
    message: AliveMessage
    done: bool


# 获取全局配置实例
alive_config = AliveConfig()

aliveMessages = []


def ollama_gen(content: str, push_tts):
    # 设置要请求的参数
    newMessage = AliveMessage(role="user", content=content)
    aliveMessages.append(newMessage)
    params = {
        "model": "qwen2.5:3b",
        "messages": [
            message.model_dump() for message in aliveMessages
        ],  # 转换为字典列表
    }
    print("params: ", params)
    ollama_api_url = alive_config.get("llm")["ollama"]["ollama_api"]

    full_response = ""
    txt = ""
    index = 1
    # 将请求转发到 Ollama API，并获取流式响应
    llm_response = requests.post(
        ollama_api_url,
        json=params,  # 将 Pydantic 模型转换为字典
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
                full_response += data["message"]["content"]
                if txt.count("。") >= 2:
                    # 根据句号分段进行 TTS 生成
                    curr, next = split_by_period(txt)
                    print(curr)
                    push_tts(curr, index)
                    index += 1
                    txt = next
    response_message = AliveMessage(role="assistant", content=full_response)
    aliveMessages.append(response_message)


def split_by_period(txt: str):
    # 找到最后一个句号的位置
    last_period_index = txt.rfind("。")

    # 根据最后一个句号的位置分割字符串
    if last_period_index != -1:
        part1 = txt[: last_period_index + 1]  # 包含最后一个句号
        part2 = txt[last_period_index + 1 :]  # 剩余部分
    else:
        part1 = txt  # 如果没有句号，整个字符串作为第一部分
        part2 = ""  # 第二部分为空
    return part1, part2


# gen_ollama("你好")
