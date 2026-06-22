from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FullDuplexConfig:
    sample_rate: int = 16000
    chunk_size: int = 1280
    chunk_ms: int = 80

    vad_threshold: float = 0.40
    vad_hangover_frames: int = 5

    sv_threshold: float = 0.29
    interrupt_confirm_ms: int = 400
    min_sv_context_chunks: int = 5
    sv_check_every_chunks: int = 2
    max_rolling_chunks: int = 30

    min_utterance_ms: int = 500
    silence_before_smartturn_ms: int = 300
    max_silence_fallback_ms: int = 900

    assistant_intro_text: str = (
        "Xin chào, tôi đang giới thiệu về hệ thống nhà thông minh Lumi. "
        "Bạn có thể ngắt lời tôi bất cứ lúc nào nếu bạn là người dùng đích, "
        "vì hệ thống này hỗ trợ nhận diện người dùng thông minh, chỉ ngắt khi đúng giọng của bạn thôi nhé."
    )
    response_on_light_command: str = "Đã bật đèn phòng khách."
    response_default: str = "Xin lỗi, tôi không hiểu."

