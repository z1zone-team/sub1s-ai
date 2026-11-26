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
    # 시작할 때 모델을 미리 로드한다. 끄면 첫 음성 요청이 모델 로드 시간(수십 초)만큼 더 걸린다. 모델이 없는 환경과 테스트에서 끈다
    whisper_preload: bool = True
    # 한 구간에서 만들 수 있는 최대 토큰 수. 같은 말을 반복하는 환각 루프가 워커를 오래 붙잡지 않게 한다
    whisper_max_new_tokens: int = 128
    # 같은 3-gram의 반복을 막는다
    whisper_no_repeat_ngram_size: int = 3

    # Gemini Config
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    nlu_timeout_sec: float = 2.0  # 2초 SLA 타임아웃

    # 음성 인식 제공자 순서. 앞이 실패하거나 제한 시간을 넘기면 다음으로 넘어가고, 목록에 없어도 마지막에는 whisper를 쓴다
    # (2026-11-26 운영 측정: gemini 2~5초, whisper small 11~14초)
    stt_providers: str = "gemini,whisper"
    # 음성 인식과 의도 힌트에 쓸 Gemini 모델, 앞에서부터. 쿼터가 모델마다 따로라(무료 키 2.5-flash 하루 20건)
    # 앞 모델이 쿼터 초과로 쉬면 다음 모델을 쓴다. 비우면 gemini_model 하나
    gemini_stt_models: str = "gemini-2.5-flash,gemini-3.5-flash-lite"
    # Gemini 음성 인식 제한 시간. 넘기면 Whisper로 넘어가므로 Whisper 시간(약 12초)과 더해도 서버 제한(25초) 안이어야 한다
    gemini_stt_timeout_sec: float = 6.0
    # 429(쿼터 초과) 응답에 retryDelay가 없을 때 Gemini를 쉬는 시간
    gemini_quota_cooldown_sec: float = 600.0
    # 503(수요 과다) 뒤 Gemini를 쉬는 시간
    gemini_unavailable_cooldown_sec: float = 30.0
    # 받아 적은 문장에서 의도 힌트를 받는 제한 시간(서버가 규칙으로 풀지 못했을 때만 부른다)
    intent_hint_timeout_sec: float = 3.0


    # 취향 문장 임베딩 (multilingual-e5-small ONNX 양자화본과 SentencePiece). 이미지 빌드 때 이 폴더에 받는다
    preference_embedding_model_dir: str = "/app/models/e5-small"

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
