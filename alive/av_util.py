import os
import threading
import time
from pydub import AudioSegment
from pydub.playback import play


play_queue = []
queue_lock = threading.Lock()  # 用于线程安全地操作队列


def tts_play_queue():
    print("tts_play_queue start")
    while True:
        with queue_lock:
            if len(play_queue) > 0:
                curr = play_queue.pop(0)
                print("now playing: ", curr)
                tts_play(curr)
        time.sleep(1)


def push_play_queue(tts_label: str, tts_index: int):
    new_tts_file = f"{tts_label}_{tts_index}.wav"
    with queue_lock:
        play_queue.append(new_tts_file)


def play_file(file_name: str):
    print(f"Playing {file_name}")
    audio = AudioSegment.from_file(file_name)
    play(audio)


def tts_play(tts_file: str):
    # 检查文件是否存在
    if not os.path.exists(tts_file):
        print(f"File {tts_file} does not exist, skipping.")
        return

    # 创建并启动线程
    thread = threading.Thread(target=play_file, args=(tts_file,))
    thread.start()
    # 等待线程完成
    thread.join()

    # 如果是临时文件，删除它
    if tts_file.startswith("alive_temp"):
        os.remove(tts_file)
