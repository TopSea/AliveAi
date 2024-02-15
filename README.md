# AliveAi
适用于 [Alive][Alive] 的 ai 工具包。

**我对 Python 的了解不多，也就只会写几个简单的脚本，欢迎提 issue 和 pr。**

## 功能计划
- 实现和语言类大模型的语音对话；
- 对 Alive 的控制和互动。   

虽然两句话就说完了，但是实现起来还是有些难度的。   

## 流程规划
- 用 [RealtimeSTT][RealtimeSTT] 实现语音唤醒和语音识别；

- 将识别到的语音分类：如果是对 Alive 的指令就直接发送给 Alive 执行，否则将识别的内容发送给大语言模型；

- 大语言模型发送回来的内容，同时：
    - 使用 [GPT-SoVITS][GPT-SoVITS] 生成对应人物的语音；
    - 做情感分析后把指令发送给 Alive。

- 播放动作和语音。


[Alive]: https://github.com/TopSea/Alive
[RealtimeSTT]: https://github.com/KoljaB/RealtimeSTT
[GPT-SoVITS]: https://github.com/RVC-Boss/GPT-SoVITS