import re


def decode_config_from_alive(config_str: str) -> str:
    # 解码步骤
    decoded_str = (
        config_str.encode("latin-1")  # 将字符串编码为 latin-1
        .decode("unicode-escape")  # 解码为 unicode-escape
        .encode("latin-1")  # 再次编码为 latin-1
        .decode("utf-8")  # 最后解码为 utf-8
    )
    # 移除控制字符
    decoded_str = re.sub(r"[\x00-\x1f\x7f]", "", decoded_str)

    # 修正 JSON 格式
    decoded_str = decoded_str.replace("'", '"')  # 将单引号替换为双引号
    return decoded_str
