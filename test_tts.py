import threading
import time
from alive.api_llm import split_by_period
from alive.av_util import push_play_queue

from alive.local_tts import append_tts_queue, tts_task_queue


test_txt = """
及至始皇，奋六世之余烈，振长策而御宇内，吞二周而亡诸侯，履至尊而制六合，执敲扑而鞭笞天下，威振四海。
南取百越之地，以为桂林、象郡；百越之君，俯首系颈，委命下吏。乃使蒙恬北筑长城而守藩篱，却匈奴七百余里。
胡人不敢南下而牧马，士不敢弯弓而报怨。于是废先王之道，焚百家之言，以愚黔首；隳名城，杀豪杰，收天下之兵，
聚之咸阳，销锋镝，铸以为金人十二，以弱天下之民。然后践华为城，因河为池，据亿丈之城，临不测之渊，以为固。
良将劲弩守要害之处，信臣精卒陈利兵而谁何。天下已定，始皇之心，自以为关中之固，金城千里，子孙帝王万世之业也。
"""


def split_string_every_50_chars(input_string):
    # 使用列表推导式，每次取 50 个字符
    return [input_string[i : i + 50] for i in range(0, len(input_string), 50)]


test_tts_queue = split_string_every_50_chars(test_txt)


txt = ""
index = 0
for line in test_tts_queue:
    txt += line.strip()
    if txt.count("。") >= 2:
        curr, next = split_by_period(txt)
        print("line: ", curr)
        append_tts_queue(curr, index)
        index += 1
        txt = next
# 别忘了最后一句
print("line: ", txt)
append_tts_queue(txt, index)

tts_task_queue("遐蝶")

# tts_thread = threading.Thread(target=tts_task_queue, args=())
# tts_thread.start()
