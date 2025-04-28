# SDK模型下载
from modelscope import snapshot_download

# snapshot_download("iic/CosyVoice2-0.5B", local_dir="pretrained_models/CosyVoice2-0.5B")
snapshot_download(
    "iic/nlp_structbert_siamese-uninlu_chinese-base",
    local_dir="pretrained_models/nlp_structbert_siamese-uninlu_chinese-base",
)
# snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
# snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='pretrained_models/CosyVoice-300M-25Hz')
# snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
# snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
# snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
# snapshot_download(
#     "angelala00/faster-whisper-small",
#     local_dir="pretrained_models/faster-whisper-small",
# )
# pip install git+https://github.com/openai/whisper.git
