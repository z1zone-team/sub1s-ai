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

# 취향 문장 임베딩 모델(intfloat/multilingual-e5-small, MIT)을 이미지에 포함한다.
# ONNX 양자화본(약 118MB)과 원본 SentencePiece 토크나이저(약 5MB)를 리비전을 고정해 받고 SHA-256을 확인한다.
# tokenizer.json은 메모리를 약 254MB 써서 받지 않는다
ARG E5_ONNX_REV=761b726dd34fb83930e26aab4e9ac3899aa1fa78
ARG E5_SPM_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3
RUN mkdir -p /app/models/e5-small \
    && curl -fsSL -o /app/models/e5-small/model_quantized.onnx \
       "https://huggingface.co/Xenova/multilingual-e5-small/resolve/${E5_ONNX_REV}/onnx/model_quantized.onnx" \
    && curl -fsSL -o /app/models/e5-small/sentencepiece.bpe.model \
       "https://huggingface.co/intfloat/multilingual-e5-small/resolve/${E5_SPM_REV}/sentencepiece.bpe.model" \
    && echo "f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193  /app/models/e5-small/model_quantized.onnx" | sha256sum -c - \
    && echo "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865  /app/models/e5-small/sentencepiece.bpe.model" | sha256sum -c -
ENV PREFERENCE_EMBEDDING_MODEL_DIR=/app/models/e5-small

# 애플리케이션 코드 복사
COPY app ./app

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
