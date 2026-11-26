"""
음성 인식 제공자 목록(어댑터).

`settings.stt_providers` 순서대로 제공자를 시도하고, 앞 제공자가 실패하거나 제한 시간을 넘기면 다음으로 넘어간다.
- `gemini`: Gemini에 오디오를 보내 받아 적기와 의도 힌트(출발역, 도착역, 경로 모드)를 한 번에 받는다. 운영 측정 2~5초
- `whisper`: 서버 CPU의 faster-whisper. 운영(CPU 0.6코어) 측정 11~14초. 항상 마지막 대체로 둔다

Gemini가 429(쿼터 초과)나 503(수요 과다)을 주면 정해진 시간 동안 Gemini를 건너뛴다. 무료 키는 모델별 하루 요청 수가
작아(2026-11-26 확인, gemini-2.5-flash 하루 20건) 쿼터가 없는 동안 계속 부르면 매 요청이 실패를 기다리게 된다.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from app.configs.settings import settings
from app.errors.exceptions import InvalidAudioFormatException, SttProcessingException
from app.pipelines.stt_pipeline import SttPipeline, clean_transcript, stt_pipeline

logger = logging.getLogger(__name__)

ROUTE_MODES = {
    "OPTIMAL",
    "FASTEST",
    "FEWEST_TRANSFERS",
    "LEAST_WALK",
    "COMFORTABLE",
    "CHEAPEST",
    "PERSONAL",
}


@dataclass
class IntentHint:
    """Gemini가 준 의도 힌트. 서버가 역 사전과 규칙으로 다시 확인하므로 그대로 믿지 않는다"""

    origin: Optional[str] = None
    destination: Optional[str] = None
    mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"origin": self.origin, "destination": self.destination, "mode": self.mode}


@dataclass
class SpeechResult:
    transcript: str
    provider: str
    duration_sec: float = 0.0
    processing_time_sec: float = 0.0
    intent: Optional[IntentHint] = None
    # 앞에서 실패해 건너뛴 제공자와 이유 (예: ["gemini:quota"])
    fallback_from: List[str] = field(default_factory=list)


class SpeechProvider(Protocol):
    name: str

    async def transcribe(self, audio_bytes: bytes, language: str) -> SpeechResult: ...


class ProviderUnavailable(Exception):
    """이 제공자를 지금 쓸 수 없다(쉬는 중, 키 없음, 쿼터, 시간 초과). 다음 제공자로 넘어간다"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _clean_station(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    cleaned = name.strip()
    if not cleaned or cleaned.lower() in {"null", "none"}:
        return None
    return re.sub(r"역$", "", cleaned) or None


def parse_intent_hint(payload: Dict[str, Any]) -> IntentHint:
    mode = payload.get("mode")
    return IntentHint(
        origin=_clean_station(payload.get("origin")),
        destination=_clean_station(payload.get("destination")),
        mode=mode if mode in ROUTE_MODES else None,
    )


def to_wav_16k_mono(audio_bytes: bytes) -> tuple[bytes, float]:
    """앱 녹음(m4a/AAC 등)을 Gemini가 받는 16kHz 모노 16bit WAV로 바꾸고 길이(초)를 돌려준다"""
    import wave

    import av

    try:
        container = av.open(io.BytesIO(audio_bytes))
    except Exception as e:  # 형식을 읽지 못하면 다음 제공자(Whisper)도 같은 이유로 실패하므로 형식 오류로 알린다
        raise InvalidAudioFormatException(f"오디오를 읽을 수 없습니다: {e}")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    pcm = bytearray()
    with container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                pcm.extend(out.to_ndarray().tobytes())
        for out in resampler.resample(None):
            pcm.extend(out.to_ndarray().tobytes())
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(pcm))
    return buffer.getvalue(), len(pcm) / 2 / 16000


_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9.]+)s")


def cooldown_seconds_for(error: Exception) -> Optional[float]:
    """Gemini 오류에서 쉬어야 할 시간을 정한다. 쉬지 않아도 되는 오류면 None"""
    text = str(error)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        match = _RETRY_DELAY.search(text)
        delay = float(match.group(1)) if match else settings.gemini_quota_cooldown_sec
        return min(max(delay, 1.0), 24 * 3600.0)
    if "503" in text or "UNAVAILABLE" in text:
        return settings.gemini_unavailable_cooldown_sec
    return None


class GeminiSpeechProvider:
    """
    Gemini 모델 목록(`gemini_stt_models`)을 앞에서부터 쓴다. 쿼터는 모델마다 따로라, 앞 모델이 429나 503으로 쉬면
    다음 모델로 넘어간다(이 오류는 바로 돌아와 시간이 거의 들지 않는다). 시간 초과나 그 밖의 오류는 다른 모델도
    오래 걸릴 수 있어 Gemini를 그만두고 다음 제공자(Whisper)로 넘긴다. 모든 모델이 한 제한 시간을 나눠 쓴다
    """

    name = "gemini"

    def __init__(
        self,
        client: Any = None,
        clock: Callable[[], float] = time.monotonic,
        models: Optional[List[str]] = None,
    ):
        from app.clients.gemini_client import GeminiClient

        self.client = client or GeminiClient()
        self.clock = clock
        configured = [m.strip() for m in settings.gemini_stt_models.split(",") if m.strip()]
        self.models = models or configured or [settings.gemini_model]
        self.paused_until: Dict[str, float] = {}

    def _ready_models(self) -> List[str]:
        now = self.clock()
        return [m for m in self.models if self.paused_until.get(m, 0.0) <= now]

    def available(self) -> bool:
        return self.client.is_configured() and bool(self._ready_models())

    def pause(self, model: str, seconds: float, why: str) -> None:
        self.paused_until[model] = self.clock() + seconds
        logger.warning(f"[Speech] Gemini {model}을(를) {seconds:.0f}초 쉰다 ({why})")

    async def _call(self, fn: Callable[..., Any], *args: Any, timeout: float) -> tuple:
        """모델을 차례로 시도한다. (응답 또는 None, 마지막 실패 이유)를 돌려준다"""
        if not self.client.is_configured():
            return None, "no_key"
        ready = self._ready_models()
        if not ready:
            return None, "paused"
        deadline = time.monotonic() + timeout
        reason = "paused"
        for model in ready:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "timeout"
            try:
                payload = await asyncio.wait_for(
                    asyncio.to_thread(fn, *args, model=model), timeout=remaining
                )
                return (payload if isinstance(payload, dict) else None), "bad_response"
            except asyncio.TimeoutError:
                return None, "timeout"
            except Exception as e:
                cooldown = cooldown_seconds_for(e)
                if cooldown is None:
                    logger.warning(f"[Speech] Gemini {model} 실패: {e}")
                    return None, "error"
                self.pause(model, cooldown, str(e)[:80])
                text = str(e)
                reason = "quota" if ("429" in text or "RESOURCE_EXHAUSTED" in text) else "unavailable"
        return None, reason

    async def transcribe(self, audio_bytes: bytes, language: str) -> SpeechResult:
        if not self.client.is_configured():
            raise ProviderUnavailable("no_key")
        if not self._ready_models():
            raise ProviderUnavailable("paused")
        started = time.time()
        wav, duration = await asyncio.to_thread(to_wav_16k_mono, audio_bytes)
        payload, reason = await self._call(
            self.client.transcribe_voice_sync,
            wav,
            "audio/wav",
            language,
            timeout=settings.gemini_stt_timeout_sec,
        )
        if payload is None:
            raise ProviderUnavailable(reason)
        if not isinstance(payload.get("transcript"), str):
            raise ProviderUnavailable("bad_response")
        return SpeechResult(
            transcript=clean_transcript(payload["transcript"]),
            provider=self.name,
            duration_sec=round(duration, 2),
            processing_time_sec=round(time.time() - started, 3),
            intent=parse_intent_hint(payload),
        )

    async def intent_hint(self, text: str) -> Optional[IntentHint]:
        """받아 적은 문장에서 의도 힌트를 받는다. 쓸 수 없으면(쉬는 중, 키 없음, 오류) None"""
        cleaned = text.strip()
        if not cleaned or not self.available():
            return None
        payload, reason = await self._call(
            self.client.intent_hint_sync, cleaned, timeout=settings.intent_hint_timeout_sec
        )
        if payload is None:
            logger.info(f"[Speech] Gemini 의도 힌트 없음 ({reason})")
            return None
        return parse_intent_hint(payload)


class WhisperSpeechProvider:
    name = "whisper"

    def __init__(self, stt: Optional[SttPipeline] = None):
        self.stt = stt or stt_pipeline

    async def transcribe(self, audio_bytes: bytes, language: str) -> SpeechResult:
        # Whisper는 동기이고 CPU를 오래 쓴다. 이벤트 루프를 막지 않게 스레드로 넘긴다
        res = await asyncio.to_thread(self.stt.transcribe, audio_bytes=audio_bytes, language=language)
        return SpeechResult(
            transcript=res.transcript,
            provider=self.name,
            duration_sec=getattr(res, "duration_sec", 0.0),
            processing_time_sec=getattr(res, "processing_time_sec", 0.0),
        )


class SpeechPipeline:
    def __init__(self, providers: List[SpeechProvider]):
        if not providers:
            raise ValueError("음성 인식 제공자가 하나는 있어야 한다")
        self.providers = providers

    @classmethod
    def from_settings(
        cls,
        stt: Optional[SttPipeline] = None,
        gemini: Optional[GeminiSpeechProvider] = None,
    ) -> "SpeechPipeline":
        names = [n.strip().lower() for n in settings.stt_providers.split(",") if n.strip()]
        providers: List[SpeechProvider] = []
        for name in names:
            if name == "gemini":
                providers.append(gemini or shared_gemini_provider())
            elif name == "whisper":
                providers.append(WhisperSpeechProvider(stt))
            else:
                logger.warning(f"[Speech] 알 수 없는 제공자 무시: {name}")
        # 어떤 설정이어도 마지막에는 Whisper가 있게 한다(외부 서비스가 모두 안 될 때)
        if not any(p.name == "whisper" for p in providers):
            providers.append(WhisperSpeechProvider(stt))
        return cls(providers)

    async def transcribe(self, audio_bytes: bytes, language: str = "ko") -> SpeechResult:
        if not audio_bytes:
            raise InvalidAudioFormatException("오디오 데이터가 비어 있습니다.")
        skipped: List[str] = []
        last_error: Optional[Exception] = None
        for index, provider in enumerate(self.providers):
            is_last = index == len(self.providers) - 1
            try:
                result = await provider.transcribe(audio_bytes, language)
                result.fallback_from = skipped
                if skipped:
                    logger.info(f"[Speech] {provider.name}로 처리 (건너뜀: {', '.join(skipped)})")
                return result
            except ProviderUnavailable as e:
                skipped.append(f"{provider.name}:{e.reason}")
                last_error = e
            except InvalidAudioFormatException:
                raise
            except Exception as e:
                if is_last:
                    raise
                logger.warning(f"[Speech] {provider.name} 실패, 다음 제공자로: {e}")
                skipped.append(f"{provider.name}:error")
                last_error = e
        raise SttProcessingException(f"음성 인식 제공자를 모두 쓸 수 없습니다: {last_error}")


_shared_gemini: Optional[GeminiSpeechProvider] = None


def shared_gemini_provider() -> GeminiSpeechProvider:
    """쉬는 시간(쿼터)을 워커와 HTTP 경로가 함께 쓰도록 프로세스에 하나만 둔다"""
    global _shared_gemini
    if _shared_gemini is None:
        _shared_gemini = GeminiSpeechProvider()
    return _shared_gemini
