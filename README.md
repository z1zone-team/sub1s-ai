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
