import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "sub1s-ai"
    app_env: str = "development"
    debug: bool = True

    # Server Config
    ai_host: str = "0.0.0.0"
    ai_port: int = 8000

    # Whisper STT Config
    whisper_model_size: str = "small"  # tiny, base, small, medium (실기기 녹음으로 비교해 small로 정함)
    whisper_device: str = "cpu"       # cpu, cuda
    whisper_compute_type: str = "int8"  # int8, float16, float32
    # CPU 한도(운영 컨테이너 0.6코어)보다 스레드가 많으면 스로틀링으로 오히려 느려진다 (small 기준 9.0초 -> 5.0초)
    whisper_cpu_threads: int = 1
    # 한 구간에서 만들 수 있는 최대 토큰 수. 같은 말을 반복하는 환각 루프가 워커를 오래 붙잡지 않게 한다
    whisper_max_new_tokens: int = 128
    # 같은 3-gram의 반복을 막는다
    whisper_no_repeat_ngram_size: int = 3

    # Gemini Config
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    nlu_timeout_sec: float = 2.0  # 2초 SLA 타임아웃

    # Edge-TTS Config
    tts_voice: str = "ko-KR-SunHiNeural"  # Microsoft 신경망 한국어 여성 음성
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"

    # Valkey / Redis / BullMQ Config
    valkey_host: str = "localhost"
    valkey_port: int = 6379
    valkey_password: Optional[str] = None
    voice_queue_name: str = "subway-voice"
    enable_bullmq_worker: bool = False
    # 동시에 처리할 큐 작업 수. Whisper는 CPU를 오래 써서 1 vCPU 환경에서는 1이 안전하다
    voice_worker_concurrency: int = 1

    # Station Data Path
    subway_stations_file: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "subway_stations.json",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
