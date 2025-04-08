# AliveAi
完全本地部署的可语音唤醒的 AI 助手。   
适配了 [Alive][Alive] 。

**我对 Python 的了解不多，也就只会写几个简单的脚本，代码可能写的很烂，特别是多线程部分。欢迎提 issue 和 pr。**

## 上游依赖
**需要 Ollama API 服务** 

## 流程规划
- [x] 用 [RealtimeSTT] 实现语音唤醒和语音识别
- [ ] 将识别到的语音分类：如果是指令就直接执行，否则将识别的内容发送给大语言模型
- [x] 连接 [Ollama] 服务器，生成对话
- [x] 使用 [Cosyvoice] 将文本转成语音并播放



[Alive]: https://github.com/TopSea/Alive
[RealtimeSTT]: https://github.com/KoljaB/RealtimeSTT
[Ollama]: https://github.com/ollama/ollama
[Cosyvoice]: https://github.com/FunAudioLLM/CosyVoice