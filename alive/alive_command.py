import json
import hanlp

hanlp.pretrained.sts.ALL  # 语义识别
sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)

# 读取命令
with open("commands.json", "r", encoding="utf-8") as file:
    data = json.load(file)
commands_label = [command["label"] for command in data["commands"]]


def create_command(words: str):
    matchs = []

    print(commands_label)
    for command in commands_label:
        matchs.append((command, words))
    return matchs


def match_and_exec(matchs: list[str]):
    results = sts(matchs)
    print(results)
    final = 0
    for result in results:
        if result > final:
            final = result
    index = results.index(final) if final > 0.6 else -1
    if index >= 0:
        commands = data["commands"]
        return exec_command(commands[index]["exec"], commands[index]["execType"])
    return False


def exec_command(exec: str, execType: str):
    print("exec: ", exec, "execType: ", execType)
    return True
