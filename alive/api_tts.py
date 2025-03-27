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
from api_llm import split_by_period

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


# txt = "春日清晨，老街深处飘来阵阵豆香。三代传承的手艺，将金黄的豆浆熬制成最纯粹的味道。一碗温热的豆腐脑，不仅是早餐，更是儿时难忘的记忆，是岁月沉淀的生活智慧。"
# cosy_tts_gen(txt, "alive_temp", 0)
test_txt = """
秦孝公据崤函之固，拥雍州之地，君臣固守以窥周室，有席卷天下，包举宇内，囊括四海之意，并吞八荒之心。
当是时也，商君佐之，内立法度，务耕织，修守战之具；外连衡而斗诸侯。于是秦人拱手而取西河之外。

孝公既没，惠文、武、昭襄蒙故业，因遗策，南取汉中，西举巴蜀，东割膏腴之地，北收要害之郡。诸侯恐惧，会盟而
谋弱秦，不爱珍器重宝肥饶之地，以致天下之士，合从缔交，相与为一。当此之时，齐有孟尝，赵有平原，楚有春
申，魏有信陵。此四君者，皆明智而忠信，宽厚而爱人，尊贤而重士，约从离衡，兼韩、魏、燕、楚、齐、赵、宋
、卫、中山之众。于是六国之士，有甯越、徐尚、苏秦、杜赫之属为之谋，齐明、周最、陈轸、召滑、楼缓、翟景
、苏厉、乐毅之徒通其意，吴起、孙膑、带佗、倪良、王廖、田忌、廉颇、赵奢之伦制其兵。尝以十倍之地，百万
之众，叩关而攻秦。秦人开关延敌，九国之师，逡巡而不敢进。秦无亡矢遗镞之费，而天下诸侯已困矣。于是从散
约败，争割地而赂秦。秦有余力而制其弊，追亡逐北，伏尸百万，流血漂橹。因利乘便，宰割天下，分裂山河。强
国请服，弱国入朝。延及孝文王、庄襄王，享国之日浅，国家无事。

及至始皇，奋六世之余烈，振长策而御宇内，吞二周而亡诸侯，履至尊而制六合，执敲扑而鞭笞天下，威振四海。
南取百越之地，以为桂林、象郡；百越之君，俯首系颈，委命下吏。乃使蒙恬北筑长城而守藩篱，却匈奴七百余里。
胡人不敢南下而牧马，士不敢弯弓而报怨。于是废先王之道，焚百家之言，以愚黔首；隳名城，杀豪杰，收天下之兵，
聚之咸阳，销锋镝，铸以为金人十二，以弱天下之民。然后践华为城，因河为池，据亿丈之城，临不测之渊，以为固。
良将劲弩守要害之处，信臣精卒陈利兵而谁何。天下已定，始皇之心，自以为关中之固，金城千里，子孙帝王万世之业也。

始皇既没，余威震于殊俗。然陈涉瓮牖绳枢之子，氓隶之人，而迁徙之徒也；才能不及中人，非有仲尼、墨翟之贤，
陶朱、猗顿之富；蹑足行伍之间，而倔起阡陌之中，率疲弊之卒，将数百之众，转而攻秦，斩木为兵，揭竿为旗，
天下云集响应，赢粮而景从。山东豪俊遂并起而亡秦族矣。

且夫天下非小弱也，雍州之地，崤函之固，自若也。陈涉之位，非尊于齐、楚、燕、赵、韩、魏、宋、卫、中山之君也；
锄櫌棘矜，非铦于钩戟长铩也；谪戍之众，非抗于九国之师也；深谋远虑，行军用兵之道，非及乡时之士也。
然而成败异变，功业相反，何也？试使山东之国与陈涉度长絜大，比权量力，则不可同年而语矣。
然秦以区区之地，致万乘之势，序八州而朝同列，百有余年矣；然后以六合为家，崤函为宫；
一夫作难而七庙隳，身死人手，为天下笑者，何也？仁义不施而攻守之势异也。
"""


def split_string_every_50_chars(input_string):
    # 使用列表推导式，每次取 50 个字符
    return [input_string[i : i + 50] for i in range(0, len(input_string), 50)]


test_tts_queue = split_string_every_50_chars(test_txt)

tts_queue = []


def tts_gen_queue():
    print("tts_queue start")
    while True:
        if len(tts_queue) > 0:
            print("tts_queue running")
            txt, index = tts_queue.pop(0)
            push_play_queue("alive_", index)
            # cosy_tts_gen(txt, "alive_", index)

        time.sleep(1)


txt = ""
index = 0
for line in test_tts_queue:
    txt += line.strip()
    if txt.count("。") >= 2:
        curr, next = split_by_period(txt)
        print("line: ", curr)
        index += 1
        txt = next


# tts_thread = threading.Thread(target=tts_play_queue, args=())
# tts_thread.daemon = True  # 设置为守护线程，主程序退出时自动终止
# tts_thread.start()
# tts_gen_queue()
