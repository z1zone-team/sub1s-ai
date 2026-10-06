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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
