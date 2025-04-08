from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from cosyvoice.cli.cosyvoice import CosyVoice2
from fastapi import FastAPI, Form
import numpy as np
import argparse
import uvicorn
import torch
import os


app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/cosy_stream")
async def inference_instruct2(
    tts_text: str = Form(),
    instruct_text: str = Form(),
    speaker: str = Form(),
    speed: float = Form(),
):
    prompt_speech_16k = load_voice_data(speaker)
    if prompt_speech_16k is None:
        return {"error": "预训练音色文件中缺少audio_ref数据！"}, 500
    model_output = cosyvoice.inference_instruct2(
        tts_text, instruct_text, prompt_speech_16k, speed=speed
    )
    return StreamingResponse(generate_data(model_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=20181)
    args = parser.parse_args()

    cosyvoice = CosyVoice2("models/CosyVoice2-0.5B")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
