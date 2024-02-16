if __name__ == '__main__':

    from RealtimeSTT import AudioToTextRecorder
    import requests


    def on_wakeword_detection_start():
        print("\non_wakeword_detection_start\n")

    def on_recording_start():
        # 开始录制指令
        print("\non_recording_start\n")

    def on_vad_detect_start():
        print("\non_vad_detect_start\n")

    def on_transcription_start():
        print("\non_transcription_start\n")

        
    def text_detected(text):
        # 录制到的指令
        print("\ntext: ", text)

    recorder_model = "large-v2"
    language = "zh"

    recorder = AudioToTextRecorder(
        model=recorder_model,
        language=language,
        wake_words="jarvis",
        silero_use_onnx=False,
        spinner=True,
        silero_sensitivity=0.2,
        webrtc_sensitivity=3,
        on_recording_start=on_recording_start,
        on_vad_detect_start=on_vad_detect_start,
        on_wakeword_detection_start=on_wakeword_detection_start,
        on_transcription_start=on_transcription_start,
        post_speech_silence_duration=0.4, 
        min_length_of_recording=0.3, 
        min_gap_between_recordings=0.01, 
        enable_realtime_transcription = True,
        realtime_processing_pause = 0.01, 
        realtime_model_type = "tiny",
        on_realtime_transcription_stabilized=text_detected
    )

    while(True):
        recorder.text()
