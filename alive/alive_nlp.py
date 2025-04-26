import json
from modelscope.pipelines import pipeline

semantic_cls = pipeline(
    "rex-uninlu",
    model="pretrained_models/nlp_deberta_rex-uninlu_chinese-base",
)


def emotion_classify(txt: str):
    classify = semantic_cls(
        input=f"[CLASSIFY]{txt}",
        schema={
            "兴奋": None,
            "开心": None,
            "失落": None,
            "哀伤": None,
            "愤怒": None,
            "惊讶": None,
            "无情绪": None,
        },
    )
    return json.dumps(classify)
