# Sub1S AI Worker Service (`sub1s-ai`)

FastAPI 기반의 고속 STT(음성 인식), Gemini NLU(의도 분석 및 슬롯 필링) 파이프라인 워커 서비스입니다.

---

## 🚀 기술 스택
- **Python**: 3.10+
- **FastAPI**: 0.110+ (비동기 엔드포인트 & Pydantic V2)
- **STT**: `faster-whisper` (int8 양자화, 기본 모델 `small`, 1 vCPU 환경 최적화)
- **NLU / 브리핑**: Google Gemini API (Single-Call 구조 & 0-토큰 가드레일)
- **Queue**: Valkey / BullMQ 비동기 큐

---

## 📨 BullMQ 작업 형식 (`subway-voice` 큐)

서버(Node BullMQ)가 큐에 작업을 넣고 이 워커(Python `bullmq`)가 처리해 결과를 돌려줍니다. 큐 이름은 `VOICE_QUEUE_NAME`(기본 `subway-voice`), 동시에 처리하는 작업 수는 `VOICE_WORKER_CONCURRENCY`(기본 1, Whisper가 CPU를 오래 쓰므로 1 vCPU에서는 1이 안전)입니다. 워커는 `ENABLE_BULLMQ_WORKER=true`일 때 FastAPI와 같은 프로세스에서 켜집니다.

| 작업(`action`) | 입력 `data` | 결과(`returnvalue`) |
| :--- | :--- | :--- |
| `stt` (`transcribe`) | `{ action: "stt", audioBase64: string }` (`audio_base64`도 받음) | `{ transcript: string, durationSec: number }` |
| `nlu` (`intent`, `voice_process`) | `{ action: "nlu", transcript: string, preferredMode?: RouteMode }` | 의도 분석 결과(`originStationName`, `destStationName`, `preferredMode` 등) |
| `briefing` (`voice_summary`) | `{ action: "briefing", originStationName, destStationName }` | 브리핑 결과 |

- **오류**: 입력이 비어 있거나 5MB를 넘는 등 잘못된 오디오는 예외가 되어 작업이 **실패(failed)** 합니다. `audioBase64`가 아예 없으면 작업은 **완료**되고 결과가 `{ "error": "Missing audio_base64 in payload" }`입니다. 호출하는 쪽은 두 경우를 모두 처리해야 합니다.
- **STT는 스레드에서 실행**됩니다(Whisper가 동기이고 CPU를 오래 써서 이벤트 루프를 막지 않도록).

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
├── configs/          # pydantic-settings 환경변수 설정
├── dtos/             # 요청/응답 Pydantic 스키마
├── errors/           # 커스텀 예외 및 글로벌 에러 핸들러
├── guardrails/       # [devine_ai 패턴] 0-토큰 역명/모드 가드레일
├── pipelines/        # STT, NLU 핵심 파이프라인
├── routes/           # FastAPI 라우트 핸들러
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
