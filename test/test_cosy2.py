import os
import sys
import torchaudio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/..".format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

cosyvoice = CosyVoice2(
    f"../pretrained_models/CosyVoice2-0.5B",
    load_jit=False,
    load_trt=False,
    fp16=False,
    use_flow_cache=True,
)
prompt_speech_16k = load_wav(f"../asset/布洛妮娅.mp3", 16000)

for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        "今天的工作都完成啦，没什么事，就早点回家吧。",
        "今天的工作都完成啦，没什么事，就早点回家吧。",
        prompt_speech_16k,
        stream=True,
    )
):
    torchaudio.save(
        "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
    )
