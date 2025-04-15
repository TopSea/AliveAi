import json
import os
import threading

import requests


class AliveConfig:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_file="alive_ai_config.json"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.config_file = config_file
                cls._instance.settings = cls._instance.load_config()
        return cls._instance

    def load_config(self):
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return json.load(f)
            pass
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {self.config_file} not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing config file: {e}")

    def get(self, key, default=None):
        with self._lock:
            return self.settings.get(key, default)

    def update(self, new_settings):
        with self._lock:
            self.settings.update(new_settings)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)


def initialize_config_file(config_file: str):
    # 不使用本地文件，直接从 Alive 请求配置
    with open(config_file, "w", encoding="utf-8") as f:
        response = requests.request(
            "POST",
            "http://localhost:20177/query/ai_config",
        )
        # 检查响应是否成功
        if response.status_code != 200:
            raise Exception(f"请求失败，状态码：{response.status_code}")
        f.write(response.text)
    print(f"配置文件 {config_file} 已创建，并写入了 Alive_AI 配置。")
