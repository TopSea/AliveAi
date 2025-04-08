import os
import sys

import torch

sys.path.append("third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


cosyvoice = CosyVoice2(
    "models/CosyVoice2-0.5B",
    load_jit=False,
    load_trt=False,
    fp16=False,
)


def load_voice_data(speaker):
    """load voice data"""
    voice_path = f"./models/tts_voices/{speaker}.pt"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(voice_path):
            return None
        voice_data = torch.load(voice_path, map_location=device)
        return voice_data.get("audio_ref")
    except Exception as e:
        raise ValueError(f"Load voice file failed: {e}")


# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_voice_data("布洛妮娅")
for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        "",
        prompt_speech_16k,
        stream=False,
    )
):
    torchaudio.save(
        "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
    )
