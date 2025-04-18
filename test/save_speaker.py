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
    use_flow_cache=False,
)
prompt_speech_16k = load_wav(f"../asset/芙宁娜.mp3", 16000)

# save zero_shot spk for future usage
assert (
    cosyvoice.add_zero_shot_spk(
        "茶会是淑女的必修课。如果你想学习茶会礼仪的话，我可以教你喔。",
        prompt_speech_16k,
        "芙宁娜",
    )
    is True
)
for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        "茶会是淑女的必修课。如果你想学习茶会礼仪的话，我可以教你喔。",
        "",
        "",
        zero_shot_spk_id="芙宁娜",
        stream=False,
    )
):
    torchaudio.save(
        "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
    )
cosyvoice.save_spkinfo()
