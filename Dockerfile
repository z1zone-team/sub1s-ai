FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 (faster-whisper용 ffmpeg, 헬스체크용 curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Whisper 모델을 이미지에 포함한다. 컨테이너가 시작할 때마다 내려받으면(small은 약 460MB)
# 헬스체크 시작 유예 시간을 넘길 수 있고, 배포마다 같은 파일을 다시 받게 된다.
# 모델을 바꾸려면 빌드 인자 WHISPER_MODEL_SIZE와 같은 값을 환경 변수로 쓴다
ARG WHISPER_MODEL_SIZE=small
ENV WHISPER_MODEL_SIZE=${WHISPER_MODEL_SIZE}
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL_SIZE}', device='cpu', compute_type='int8')"

# 애플리케이션 코드 복사
COPY app ./app

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
