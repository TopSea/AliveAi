import requests, json
import base64
from io import BytesIO
import time


# Alive 的 ip 地址
base_url = "http://10.158.197.102:20177"


def format_json(txt: str):
    js=json.loads(txt)
    fromatted = json.dumps(js, sort_keys=True, indent=4, separators=(',', ':'))
    print(fromatted)

    
# Hello Alive
def hello_alive():
    url = "/"
    True_url = base_url+url
    response = requests.get(True_url)
    print(response)

def change_motion():
    url = "/change/motion"
    True_url = base_url+url
    payload = {
        "mode": "mmd",                      # 在什么模式下。目前只有 mmd 实现了
        "motion_name": "dance1",            # 动作名称
        "interrupt": False,                 # 是否打断当前的动作
    }
    response = requests.post(True_url, json=payload)
    reply = response.text
    format_json(reply)
 
if __name__ == '__main__':
    change_motion()