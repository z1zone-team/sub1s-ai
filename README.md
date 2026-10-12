# Sub1S AI Worker Service (`sub1s-ai`)

FastAPI 기반의 고속 STT(음성 인식), Gemini NLU(의도 분석 및 슬롯 필링), 그리고 Edge-TTS(자연어 음성 합성) 파이프라인 워커 서비스입니다.

---

## 🚀 기술 스택
- **Python**: 3.10+
- **FastAPI**: 0.110+ (비동기 엔드포인트 & Pydantic V2)
- **STT**: `faster-whisper` (int8 양자화, 1 vCPU 환경 최적화)
- **NLU / 브리핑**: Google Gemini API (Single-Call 구조 & 0-토큰 가드레일)
- **TTS**: `edge-tts` (Microsoft 신경망 음성 `ko-KR-SunHiNeural`)
- **Queue**: Valkey / BullMQ 비동기 큐

---

## 📨 BullMQ 작업 형식 (`subway-voice` 큐)

서버(Node BullMQ)가 큐에 작업을 넣고 이 워커(Python `bullmq`)가 처리해 결과를 돌려줍니다. 큐 이름은 `VOICE_QUEUE_NAME`(기본 `subway-voice`), 동시에 처리하는 작업 수는 `VOICE_WORKER_CONCURRENCY`(기본 1, Whisper가 CPU를 오래 쓰므로 1 vCPU에서는 1이 안전)입니다. 워커는 `ENABLE_BULLMQ_WORKER=true`일 때 FastAPI와 같은 프로세스에서 켜집니다.

| 작업(`action`) | 입력 `data` | 결과(`returnvalue`) |
| :--- | :--- | :--- |
| `stt` (`transcribe`) | `{ action: "stt", audioBase64: string }` (`audio_base64`도 받음) | `{ transcript: string, durationSec: number }` |
| `nlu` (`intent`, `voice_process`) | `{ action: "nlu", transcript: string, preferredMode?: RouteMode }` | 의도 분석 결과(`originStationName`, `destStationName`, `preferredMode` 등) |
| `tts` (`synthesize_tts`) | `{ action: "tts", text: string, voice?: string }` | `{ audioBase64: string, bytesLength: number }` |
| `briefing` (`voice_summary`) | `{ action: "briefing", originStationName, destStationName }` | 브리핑 결과 |

- **오류**: 입력이 비어 있거나 5MB를 넘는 등 잘못된 오디오는 예외가 되어 작업이 **실패(failed)** 합니다. `audioBase64`가 아예 없으면 작업은 **완료**되고 결과가 `{ "error": "Missing audio_base64 in payload" }`입니다. 호출하는 쪽은 두 경우를 모두 처리해야 합니다.
- **STT는 스레드에서 실행**됩니다(Whisper가 동기이고 CPU를 오래 써서 이벤트 루프를 막지 않도록).
- 큐는 오디오를 base64로 담으므로 Valkey 메모리를 씁니다(운영 Valkey 80MB, `noeviction`). 보내는 쪽에서 크기를 제한하고 끝난 작업은 짧게 보관해야 합니다.

## 📂 프로젝트 구조
```text
app/
├── configs/          # pydantic-settings 환경변수 설정
├── dtos/             # 요청/응답 Pydantic 스키마
├── errors/           # 커스텀 예외 및 글로벌 에러 핸들러
├── guardrails/       # [devine_ai 패턴] 0-토큰 역명/모드 가드레일
├── pipelines/        # STT, NLU, TTS 핵심 파이프라인
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
