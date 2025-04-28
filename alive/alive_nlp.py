import json
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline

semantic_cls = pipeline(
    Tasks.siamese_uie,
    model="pretrained_models/nlp_structbert_siamese-uninlu_chinese-base",
)


def emotion_classify(txt: str):
    classify = semantic_cls(
        input=f"无情绪,兴奋,开心,失落,哀伤,愤怒,惊讶|{txt}。", schema={"情感分类": None}
    )
    print(classify)
    return json.dumps(classify)
