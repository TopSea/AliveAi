import threading
import time
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2
from alive.alive_config import AliveConfig
import base64
import os
from pydub import AudioSegment

cosyvoice = CosyVoice2(
    "pretrained_models/CosyVoice2-0.5B", load_jit=False, load_trt=False, fp16=True
)
alive_config = AliveConfig()
tts_done = False
tts_queue = []

prompt_speech_16k = None


def set_prompt_speech_16k(speaker: str):
    global prompt_speech_16k
    prompt_speech_16k = load_wav(f"./asset/{speaker}.mp3", 16000)


def tts_task_queue_zero(speaker: str):
    global tts_done, tts_queue, prompt_speech_16k
    os.makedirs("./asset/temp", exist_ok=True)
    if not prompt_speech_16k:
        prompt_speech_16k = load_wav(f"./asset/{speaker}.mp3", 16000)
    while True:
        if len(tts_queue) > 0:
            txt, index = tts_queue.pop(0)
            if len(txt) > 0:
                generate_zero(txt, index)
            if len(tts_queue) == 0:
                time.sleep(1)
                tts_done = True
        else:
            time.sleep(0.1)


def generate_zero(txt, index):
    global prompt_speech_16k
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            tts_text=txt,
            prompt_text=alive_config.get("tts")["cosy"]["instruct_text"],
            prompt_speech_16k=prompt_speech_16k,
            stream=False,
            speed=1.0,
        )
    ):
        torchaudio.save(
            f"./asset/temp/{index}.wav", j["tts_speech"], cosyvoice.sample_rate
        )
    # running_tasks -= 1


def append_tts_queue(txt, index):
    tts_queue.append((txt, index))


def is_tts_done():
    return tts_done


def set_tts_start():
    global tts_done
    tts_done = False


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert (
            sample_rate > target_sr
        ), "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


def check_audio():
    """return if has generated audio file. return the first one"""
    audio_path = "./asset/temp"
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
        return None

    contents = os.listdir(audio_path)
    files = [
        item for item in contents if os.path.isfile(os.path.join(audio_path, item))
    ]

    if not files:
        return None

    files.sort()

    first_file_path = os.path.join(audio_path, files[0])
    print(f"first file is: {first_file_path}")
    return first_file_path
