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
running_tasks = 0


def tts_task_queue(speaker: str):
    global tts_done, tts_queue
    os.makedirs("./asset/temp", exist_ok=True)
    while True:
        if len(tts_queue) > 0:
            txt, index = tts_queue.pop(0)
            if len(txt) > 0:
                print("tts_queue running: ", txt)
                generate(txt, index, speaker)
            if len(tts_queue) == 0:
                time.sleep(1)
                tts_done = True
        else:
            time.sleep(0.1)


def tts_task_queue_zero(speaker: str):
    global tts_done, tts_queue
    os.makedirs("./asset/temp", exist_ok=True)
    prompt_speech_16k = load_wav(f"./asset/{speaker}.mp3", 16000)
    while True:
        if len(tts_queue) > 0:
            txt, index = tts_queue.pop(0)
            if len(txt) > 0:
                generate_zero(txt, index, prompt_speech_16k)
            if len(tts_queue) == 0:
                time.sleep(1)
                tts_done = True
        else:
            time.sleep(0.1)


def tts_mul_thread():
    """
    multi thread but has some problem: 1.wav may generated faster than 0.wav.
    good for non stream task.
    """
    global tts_done, tts_queue, running_tasks
    os.makedirs("./asset/temp", exist_ok=True)
    prompt_speech_16k = load_wav("./asset/Broniya.mp3", 16000)
    max_task = 2
    while True:
        if len(tts_queue) > 0:
            if running_tasks < max_task:
                running_tasks += 1
                txt, index = tts_queue.pop(0)
                print("tts_queue running: ", txt)

                tts_thread = threading.Thread(
                    target=generate, args=(txt, index, prompt_speech_16k), daemon=True
                )
                tts_thread.start()
            else:
                # print("max tasks reached: ", txt)
                time.sleep(1)
            tts_done = len(tts_queue) == 0
        else:
            time.sleep(1)


def generate(txt, index, speaker):
    for i, j in enumerate(
        cosyvoice.inference_sft(
            tts_text=txt,
            spk_id=speaker,
            stream=False,
            speed=1.0,
        )
    ):
        torchaudio.save(
            f"./asset/temp/{index}.wav", j["tts_speech"], cosyvoice.sample_rate
        )
    # running_tasks -= 1


def generate_zero(txt, index, prompt_speech_16k):
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            tts_text=txt,
            prompt_text="原来是你在散播消息，怪不得我怎么查也没查到是谁在推动舆论。",
            prompt_speech_16k=prompt_speech_16k,
            stream=False,
            speed=1.1,
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


def load_speaker_data(speaker):
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


def check_and_encode():
    """return if has generated audio file in base64 code. return the first one"""
    first_file_path = check_audio()
    if not first_file_path:
        return None, None
    return read_audio_file(first_file_path)


def read_audio_file(file_path: str) -> str:
    audio = AudioSegment.from_file(file_path)
    audio_bytes = audio.raw_data
    base64_data = base64.b64encode(audio_bytes).decode("utf-8")
    os.remove(file_path)
    return base64_data
