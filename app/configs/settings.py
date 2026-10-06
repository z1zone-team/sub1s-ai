import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "sub1s-ai"
    app_env: str = "development"
    debug: bool = True

    # Server Config
    ai_host: str = "0.0.0.0"
    ai_port: int = 8000

    # Whisper STT Config
    whisper_model_size: str = "tiny"  # tiny, base, small
    whisper_device: str = "cpu"       # cpu, cuda
    whisper_compute_type: str = "int8"  # int8, float16, float32

    # Gemini Config
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    nlu_timeout_sec: float = 2.0  # 2초 SLA 타임아웃

    # Edge-TTS Config
    tts_voice: str = "ko-KR-SunHiNeural"  # Microsoft 신경망 한국어 여성 음성
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"

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
