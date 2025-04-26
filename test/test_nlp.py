from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from alive.alive_nlp import emotion_classify


# semantic_cls = pipeline(
#     "rex-uninlu",
#     model="../pretrained_models/nlp_deberta_rex-uninlu_chinese-base",
# )

# 事件抽取 {事件类型（事件触发词）: {参数类型: None}}
# test = semantic_cls(
#     input="7月28日，天津泰达在德比战中以0-1负于天津天海。",
#     schema={
#         "胜负(事件触发词)": {"时间": None, "败者": None, "胜者": None, "赛事名称": None}
#     },
# )
# print(f"test: {test}")

# # 属性情感抽取 {属性词: {情感词: None}}
# emotion_classify = semantic_cls(
#     input="很满意，音质很好，发货速度快，值得购买",
#     schema={
#         "属性词": {
#             "情感词": None,
#         }
#     },
# )
# print(f"emotion_classify4: {emotion_classify}")

# # 允许属性词缺省，#表示缺省
# emotion_classify = semantic_cls(
#     input="#很满意，音质很好，发货速度快，值得购买",
#     schema={
#         "属性词": {
#             "情感词": None,
#         }
#     },
# )
# print(f"emotion_classify3: {emotion_classify}")

# # 支持情感分类
# emotion_classify = semantic_cls(
#     input="很满意，音质很好，发货速度快，值得购买",
#     schema={
#         "属性词": {
#             "正向情感(情感词)": None,
#             "负向情感(情感词)": None,
#             "中性情感(情感词)": None,
#         }
#     },
# )
# print(f"emotion_classify2: {emotion_classify}")

# 情感分类，正文前添加[CLASSIFY]，schema列举期望抽取的候选“情感倾向标签”；同时也支持情绪分类任务，换成相应情绪标签即可，e.g. "无情绪,积极,愤怒,悲伤,恐惧,惊奇"
input = (
    """如果你是在询问“雷猴”的具体含义或用途，但没有提供更多的背景信息，那么我之前的回答可能存在误解。请提供更多详细信息，这样我可以更准确地帮助你了解“雷猴”的含义或用途。例如：

1. “雷猴”是某个特定领域内的术语吗？
2. 它是一个游戏名称、电影名还是某种网络用语？
3. 是否有具体的上下文场景？

提供这些额外的信息将有助于我给出更为精确的回答。""",
)
classify = emotion_classify(input)
print(f"emotion_classify1: {classify}")

# 文本匹配，正文前添加[CLASSIFY]，待匹配段落按照“段落1：段落1文本；段落2：段落2文本”，schema按照“文本匹配prompt+候选标签”的形式构造
# text_match = semantic_cls(
#     input="[CLASSIFY]段落1：高分子材料与工程排名；段落2：高分子材料与工程专业的完整定义",
#     schema={"文本匹配：相似": None, "文本匹配：不相似": None},
# )
# print(f"text_match: {text_match}")
