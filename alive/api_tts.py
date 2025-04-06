import time
import requests
import torch
import torchaudio
import numpy as np
import threading
import requests
import logging

from alive.alive_config import AliveConfig
from alive.av_util import push_play_queue, tts_play, tts_play_queue
from alive.api_llm import split_by_period

# 获取全局配置实例
alive_config = AliveConfig()
prompt_sr, target_sr = 16000, 22050


def cosy_tts_gen(content: str, tts_label: str = "alive_temp", index: int = 0):
    cosy = alive_config.get("tts")["cosy"]

    payload = {
        "tts_text": content,
        "instruct_text": cosy["instruct_text"],
        "speaker": cosy["speaker"],
        "speed": cosy["speed"],
    }
    response = requests.request("GET", cosy["cosy_api"], data=payload, stream=True)

    tts_audio = b""
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r
    tts_speech = torch.from_numpy(
        np.array(np.frombuffer(tts_audio, dtype=np.int16))
    ).unsqueeze(dim=0)

    torchaudio.save(f"{tts_label}_{index}.wav", tts_speech, target_sr)

    # 添加生成的音频
    push_play_queue(tts_label, index)


def melo_tts_gen(content: str, index: int = 1):
    data = {
        "text": content,
        "speed": alive_config.get("conf_ai.tts.speed"),
        "language": alive_config.get("conf_ai.tts.language"),
        "speaker_id": alive_config.get("conf_ai.tts.speaker"),
    }
    tts_response = requests.post(alive_config.get("conf_ai.tts.tts_host"), json=data)

    # audio = AudioSegment.from_file(io.BytesIO(tts_response.content))
    # print(audio.duration_seconds)
    # play(audio)

    with open(f"{index}.wav", "wb") as f:
        f.write(tts_response.content)

    # 播放音频
    thread = threading.Thread(target=tts_play, args=())
    thread.start()
