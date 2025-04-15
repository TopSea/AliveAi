# AliveAi
可以完全本地部署的可语音唤醒的 AI 助手。   
适配了 [Alive][Alive] 。

**我对 Python 的了解不多，也就只会写几个简单的脚本，代码可能写的很烂，特别是多线程部分。欢迎提 issue 和 pr。**

## 上游依赖
**需要 Ollama API 服务** 

## 流程规划
- [x] 用 [RealtimeSTT] 实现语音唤醒和语音识别（本软件独立运行才需要，[Alive] 本身自带唤醒功能）
- [ ] 将识别到的语音分类：如果是指令就直接执行，否则将识别的内容发送给大语言模型
- [x] 连接 [Ollama] 服务器，生成对话
- [x] 使用 [Cosyvoice] 将文本转成语音并播放


## 部署运行
**我只在 Windows 11 上运行过，其他平台没有测试过，不知道会不会有问题。**  

**安装依赖**
```shell
conda create -n alive_ai -y python=3.10
conda activate alive_ai
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel

```

**运行**   
可以在 `alive_ai_config.json` 更改启动设置（现在大概只有api地址可以改 :sleepy: ）
```shell
# 先下载模型（可以更改代码，根据需要下载）
python ./test/download_models.py
# 启动 AliveAi 服务器
python app.py
# 独立运行（尚未完成）
python furina.py
```


[Alive]: https://github.com/TopSea/Alive
[RealtimeSTT]: https://github.com/KoljaB/RealtimeSTT
[Ollama]: https://github.com/ollama/ollama
[Cosyvoice]: https://github.com/FunAudioLLM/CosyVoice