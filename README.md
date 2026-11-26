# Sub1S AI Worker Service (`sub1s-ai`)

FastAPI 기반의 STT(음성 인식), 경로 선호 모델(10축), 취향 문장 임베딩, Gemini NLU(의도 분석) 서비스입니다.

---

## 🚀 기술 스택
- **Python**: 3.10+
- **FastAPI**: 0.110+ (비동기 엔드포인트 & Pydantic V2)
- **STT**: `faster-whisper` (int8 양자화, 기본 모델 `small`, 1 vCPU 환경 최적화)
- **NLU / 브리핑**: Google Gemini API (Single-Call 구조 & 0-토큰 가드레일)
- **선호 모델**: 선택한 경로와 대안의 피처로 10축 선호를 계산(조건부 로짓 가중치 학습 포함)
- **취향 임베딩**: `multilingual-e5-small` ONNX 양자화본(384차원, 정규화). 이미지 빌드 때 `/app/models/e5-small`에 받는다
- **Queue**: Valkey / BullMQ 비동기 큐

---

## 🌐 HTTP API (`/api/v1/ai`)

| Method | 경로 | 내용 | 서버 사용 |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | 상태와 Whisper 모델 | 서버 `/api/health`가 확인 |
| `POST` | `/api/v1/ai/stt` | 음성 인식(동기, 큐 `stt`와 같은 제공자 순서). 응답에 `provider`, `fallback_from`, `intent`(Gemini일 때) | 서버는 주로 아래 큐의 `stt` 작업을 쓴다 |
| `POST` | `/api/v1/ai/intent-hint` | 받아 적은 문장에서 출발지, 도착지, 경로 조건 힌트(Gemini). 쉬는 중이거나 키가 없으면 `{ available: false }` | 서버가 규칙으로 풀지 못한 문장에만 호출 |
| `POST` | `/api/v1/ai/preference-vector/v2` | 선택한 경로와 대안의 피처로 10축 선호와 관측 신뢰도 | 경로 검색 때 2초 제한으로 호출 |
| `POST` | `/api/v1/ai/preference-embedding` | 이동 목적, 경로 취향, 최근 고른 모드로 만든 취향 문장의 임베딩 | 경로 검색 때 2초 제한으로 호출(이웃 선호 2차 정렬) |
| `POST` | `/api/v1/ai/nlu` | 전사문의 의도 분석(Gemini, 실패 시 규칙) | 쓰지 않음(서버 규칙 파서 사용) |
| `POST` | `/api/v1/ai/briefing` | 경로 결과의 브리핑 문장 | 쓰지 않음(서버 규칙 문구 사용) |
| `POST` | `/api/v1/ai/voice-process` | STT와 NLU를 한 번에 | 쓰지 않음 |

계약 상세는 sub1s-docs의 `API_SPEC.md` 4장과 `ROUTE_PREFERENCE_V2.md`에 있다. 음성 합성(TTS)과 6축 선호 벡터 API는 삭제했다(PR #29, #31).

---

## 📨 BullMQ 작업 형식 (`subway-voice` 큐)

서버(Node BullMQ)가 큐에 작업을 넣고 이 워커(Python `bullmq`)가 처리해 결과를 돌려줍니다. 큐 이름은 `VOICE_QUEUE_NAME`(기본 `subway-voice`), 동시에 처리하는 작업 수는 `VOICE_WORKER_CONCURRENCY`(기본 1, Whisper가 CPU를 오래 쓰므로 1 vCPU에서는 1이 안전)입니다. 워커는 `ENABLE_BULLMQ_WORKER=true`일 때 FastAPI와 같은 프로세스에서 켜집니다.

| 작업(`action`) | 입력 `data` | 결과(`returnvalue`) |
| :--- | :--- | :--- |
| `stt` (`transcribe`) | `{ action: "stt", audioBase64: string }` (`audio_base64`도 받음) | `{ transcript, durationSec, provider: "gemini" \| "whisper", fallbackFrom?: string[], intent?: { origin, destination, mode } }` |
| `nlu` (`intent`, `voice_process`, 서버는 쓰지 않음) | `{ action: "nlu", transcript: string, preferredMode?: RouteMode }` | 의도 분석 결과(`originStationName`, `destStationName`, `preferredMode` 등) |
| `briefing` (`voice_summary`, 서버는 쓰지 않음) | `{ action: "briefing", originStationName, destStationName }` | 브리핑 결과 |

- **오류**: 입력이 비어 있거나 5MB를 넘는 등 잘못된 오디오는 예외가 되어 작업이 **실패(failed)** 합니다. `audioBase64`가 아예 없으면 작업은 **완료**되고 결과가 `{ "error": "Missing audio_base64 in payload" }`입니다. 호출하는 쪽은 두 경우를 모두 처리해야 합니다.
- **STT는 스레드에서 실행**됩니다(Whisper가 동기이고 CPU를 오래 써서 이벤트 루프를 막지 않도록).
- `intent`는 Gemini가 들은 대로 뽑은 **힌트**입니다. 서버가 역 사전과 규칙으로 다시 확인합니다.

## 🎙️ 음성 인식 제공자 (Gemini, Whisper)

`STT_PROVIDERS`(기본 `gemini,whisper`) 순서로 시도하고, 앞 제공자를 쓸 수 없으면 다음으로 넘어갑니다. 목록에 없어도 마지막에는 항상 Whisper를 씁니다(`app/pipelines/speech_pipeline.py`).

| 환경 변수 | 기본값 | 설명 |
| :--- | :-: | :--- |
| `STT_PROVIDERS` | `gemini,whisper` | `whisper`만 두면 Gemini를 쓰지 않습니다 |
| `GEMINI_STT_MODELS` | `gemini-2.5-flash,gemini-3.5-flash-lite` | 앞에서부터 쓰는 Gemini 모델. 쿼터는 모델마다 따로라 앞 모델이 429, 503으로 쉬면 다음 모델로 넘어갑니다 |
| `GEMINI_STT_TIMEOUT_SEC` | `6` | 모든 Gemini 모델이 나눠 쓰는 제한 시간. 넘기면 남은 모델을 건너뛰고 Whisper로 갑니다(Whisper 약 12초와 더해 서버 제한 25초 안) |
| `GEMINI_QUOTA_COOLDOWN_SEC` | `600` | 429 응답에 `retryDelay`가 없을 때 그 모델을 쉬는 시간. 있으면 `retryDelay`만큼 쉽니다 |
| `GEMINI_UNAVAILABLE_COOLDOWN_SEC` | `30` | 503(수요 과다) 뒤 그 모델을 쉬는 시간 |
| `INTENT_HINT_TIMEOUT_SEC` | `3` | `intent-hint` 제한 시간 |

- **Gemini에는 오디오를 16kHz 모노 WAV로 바꿔 보냅니다**(PyAV). 앱 녹음(m4a, AAC)도 그대로 받습니다
- **생각 단계를 끕니다.** 2.x는 `thinking_budget=0`, 3.x는 budget을 받지 않아(400) `thinking_level=MINIMAL`
- **운영 측정(2026-11-26, 합성 한국어 7문장, CPU 0.6코어)**: Whisper `small` 11.6~14.0초. `gemini-2.5-flash` 2.2~4.9초(중앙값 2.7초) 7/7 정확, `gemini-3.5-flash-lite` 1.6~5.8초 6/6 정확(한 건은 6초 제한 초과), m4a 1.6~1.9초. 경로 조건(빠른 길, 환승 적게, 사람 적은 길)도 모두 맞게 뽑았습니다. Whisper는 입력을 30초로 채워 인코더를 돌려 짧은 말도 11초 이상이고 `chunk_length`, `without_timestamps`, 스레드 수, float32로는 빨라지지 않았습니다. `base`는 3.6~4.6초지만 `12도역`(여의도역), `고속템이 노력해서`(고속터미널역에서)처럼 틀립니다
- **무료 키 쿼터**: `gemini-2.5-flash`는 하루 20건(`GenerateRequestsPerDayPerProjectPerModel`)이고 분당 한도도 작습니다. 쿼터가 없는 동안에는 그 모델을 건너뛰어 매 요청이 실패를 기다리지 않습니다. 쉬는 시간은 프로세스 메모리에 있어 재시작하면 처음부터 다시 시도합니다
- 무료 등급 Gemini API에 보낸 입력은 Google 서비스 개선에 쓰일 수 있습니다(약관). 음성을 외부로 보내지 않으려면 `STT_PROVIDERS=whisper`로 둡니다

## 🎙️ STT 설정 (Whisper)

| 환경 변수 | 기본값 | 설명 |
| :--- | :-: | :--- |
| `WHISPER_MODEL_SIZE` | `small` | `tiny`, `base`, `small`, `medium`. 이미지는 빌드할 때 이 모델을 내려받아 포함합니다(`docker build --build-arg WHISPER_MODEL_SIZE=base .`) |
| `WHISPER_CPU_THREADS` | `1` | 컨테이너 CPU 한도보다 스레드가 많으면 스로틀링으로 오히려 느려집니다 |
| `WHISPER_MAX_NEW_TOKENS` | `128` | 한 구간에서 만들 최대 토큰. 같은 말을 반복하는 환각 루프가 워커를 붙잡지 않게 합니다 |
| `WHISPER_NO_REPEAT_NGRAM_SIZE` | `3` | 같은 3-gram의 반복을 막습니다 |

- **모델은 실기기(SM-G981N)로 녹음한 14개를 CPU 1개, int8, 한국어로 비교해 정했습니다.** 1건(5초 안팎) 처리 시간: `tiny` 1.0초, `base` 1.8초, `small` 5.2초, `medium` 14.1초, `large-v3` 35초. 운영 컨테이너는 0.6코어라 `cpu_threads=1`에서 `small`이 약 5초, `medium` 이상은 서버의 STT 대기 제한(25초)을 넘고 메모리도 2GB를 넘어 쓸 수 없습니다. `tiny`는 `강남력`, `병지대`처럼 자주 틀립니다
- **메모리**: 서비스 컨테이너 전체(FastAPI, 워커 포함)가 대기 중 약 380MB, 요청을 처리할 때 최대 약 680MB입니다(CPU 0.6코어 제한, 요청 6건 순차 측정). 운영 컨테이너 상한 1GB로 충분합니다. 동시에 여러 요청을 처리하면 늘어납니다
- **말소리가 없으면 빈 전사(`""`)를 돌려줍니다.** 가짜 문장으로 채우지 않습니다. 모델을 불러오지 못하면 오류입니다(예전에는 가짜 문장을 돌려줬음)
- 모델 크기와 상관없이 역 이름 오인식(`명지대` -> `명시대`)은 남을 수 있습니다. 역 사전으로 바로잡는 것은 서버의 음성 의도 분석에서 합니다
- 큐는 오디오를 base64로 담으므로 Valkey 메모리를 씁니다(운영 Valkey 80MB, `noeviction`). 보내는 쪽에서 크기를 제한하고 끝난 작업은 짧게 보관해야 합니다.

## 📂 프로젝트 구조
```text
app/
├── clients/          # Gemini 클라이언트 (2초 타임아웃)
├── configs/          # pydantic-settings 환경변수 설정
├── data/             # 역 이름 사전 등 정적 데이터
├── dtos/             # 요청/응답 Pydantic 스키마
├── errors/           # 커스텀 예외 및 글로벌 에러 핸들러
├── guardrails/       # [devine_ai 패턴] 0-토큰 역명/모드 가드레일
├── pipelines/        # STT, NLU, 브리핑, 10축 선호(preference_v2, preference_logit), 취향 임베딩(preference_embedding)
├── prompts/          # NLU, 브리핑 프롬프트
├── routes/           # FastAPI 라우트 핸들러
├── workers/          # BullMQ 워커(voice_worker), 독립 실행(worker_main)
└── main.py           # 애플리케이션 진입점
tests/                # pytest 테스트 스위트
```

---

## 🧪 실행 및 테스트 방법
```bash
# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 단위/통합 테스트 실행
python -m pytest

# 로컬 개발 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
