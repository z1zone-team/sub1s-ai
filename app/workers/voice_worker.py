import asyncio
import base64
import io
import logging
from typing import Any, Dict, Optional

import redis.asyncio as aioredis
from bullmq import Job, Worker

from app.configs.settings import settings
from app.dtos.request_dto import RouteMode
from app.pipelines.briefing_pipeline import BriefingPipeline
from app.pipelines.nlu_pipeline import NluPipeline
from app.pipelines.stt_pipeline import SttPipeline, stt_pipeline
from app.pipelines.tts_pipeline import TtsPipeline

logger = logging.getLogger(__name__)


class VoiceWorker:
    """
    Valkey(Redis) BullMQ 큐 비동기 음성 처리 워커.
    - 큐 이름: settings.voice_queue_name (기본: subway-voice)
    - 지원 작업: 음성 전사(STT), 의도 분석(NLU), 음성 합성(Edge-TTS)
    - Fault-Tolerant 장애 격리: Valkey 미실행 환경에서도 크래시 없이 대기
    """

    def __init__(
        self,
        queue_name: Optional[str] = None,
        valkey_host: Optional[str] = None,
        valkey_port: Optional[int] = None,
        valkey_password: Optional[str] = None,
        concurrency: Optional[int] = None,
        stt: Optional[SttPipeline] = None,
        nlu: Optional[NluPipeline] = None,
        tts: Optional[TtsPipeline] = None,
        briefing: Optional[BriefingPipeline] = None,
    ):
        self.queue_name = queue_name or settings.voice_queue_name
        self.valkey_host = valkey_host or settings.valkey_host
        self.valkey_port = valkey_port or settings.valkey_port
        self.valkey_password = valkey_password or settings.valkey_password
        self.concurrency = max(1, concurrency or settings.voice_worker_concurrency)

        self.stt = stt or stt_pipeline
        self.nlu = nlu or NluPipeline.get_instance()
        self.tts = tts or TtsPipeline.get_instance()
        self.briefing = briefing or BriefingPipeline.get_instance()

        self._worker: Optional[Worker] = None
        self._is_running = False

    async def is_valkey_available(self) -> bool:
        """Valkey(Redis) 연결 가능 여부를 비동기 확인"""
        try:
            client = aioredis.Redis(
                host=self.valkey_host,
                port=self.valkey_port,
                password=self.valkey_password,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            )
            await client.ping()
            await client.aclose()
            return True
        except Exception:
            return False

    async def process_job(self, job: Job, token: Optional[str] = None) -> Dict[str, Any]:
        """
        BullMQ 큐 작업 처리기 (단위 테스트에서 직간접 호출 가능)
        """
        data = job.data or {}
        action = data.get("action", job.name)
        logger.info(f"[VoiceWorker] 작업 수신: jobId={job.id}, action='{action}'")

        if action in ("voice_process", "nlu", "intent"):
            transcript = data.get("transcript", "")
            mode_str = data.get("preferred_mode") or data.get("preferredMode")
            mode = (
                RouteMode(mode_str)
                if mode_str in [m.value for m in RouteMode]
                else None
            )
            res = await self.nlu.parse_intent(text=transcript, preferred_mode=mode)
            return res.model_dump(by_alias=True)

        elif action in ("stt", "transcribe"):
            audio_b64 = data.get("audio_base64") or data.get("audioBase64")
            if not audio_b64:
                return {"error": "Missing audio_base64 in payload"}
            audio_bytes = base64.b64decode(audio_b64)
            # Whisper는 동기이고 CPU를 오래 쓴다. 이벤트 루프에서 직접 부르면 전사하는 동안 같은 프로세스의
            # 헬스체크와 다른 요청이 멈추므로 스레드로 넘긴다
            stt_res = await asyncio.to_thread(self.stt.transcribe, audio_bytes=audio_bytes)
            return {
                "transcript": stt_res.transcript,
                "durationSec": stt_res.duration_sec,
            }

        elif action in ("tts", "synthesize_tts"):
            text = data.get("text", "")
            voice = data.get("voice")
            mp3_bytes = await self.tts.synthesize_mp3_bytes(text=text, voice=voice)
            return {
                "audioBase64": base64.b64encode(mp3_bytes).decode("utf-8"),
                "bytesLength": len(mp3_bytes),
            }

        elif action in ("briefing", "voice_summary"):
            origin = data.get("originStationName") or data.get("origin_station_name", "강남")
            dest = data.get("destStationName") or data.get("dest_station_name", "을지로입구")
            briefing_res = await self.briefing.generate_briefing(
                origin_station_name=origin,
                dest_station_name=dest,
            )
            return briefing_res.model_dump(by_alias=True)

        else:
            logger.warning(f"[VoiceWorker] 지원하지 않는 액션: {action}")
            return {"status": "unsupported_action", "action": action}

    async def start(self) -> bool:
        """
        워커를 시작합니다. Valkey 연결이 불가능한 경우 예외로 중단되지 않고
        경고 로깅 후 안전하게 대기 모드를 유지합니다 (Fault Tolerance).
        """
        available = await self.is_valkey_available()
        if not available:
            logger.warning(
                f"[VoiceWorker] Valkey(Redis) {self.valkey_host}:{self.valkey_port} 연결 불가. "
                f"워커는 대기 상태를 유지하며 HTTP API는 정상 동작합니다."
            )
            return False

        try:
            connection_opts = {
                "host": self.valkey_host,
                "port": self.valkey_port,
            }
            if self.valkey_password:
                connection_opts["password"] = self.valkey_password

            self._worker = Worker(
                name=self.queue_name,
                processor=self.process_job,
                opts={"connection": connection_opts, "concurrency": self.concurrency},
            )
            self._is_running = True
            logger.info(
                f"[VoiceWorker] BullMQ 워커 구동 완료. 큐: '{self.queue_name}' 수신 대기 중..."
            )
            return True
        except Exception as e:
            logger.error(f"[VoiceWorker] 워커 초기화 실패: {e}")
            return False

    async def stop(self) -> None:
        """워커를 안전하게 종료합니다."""
        if self._worker and self._is_running:
            try:
                await self._worker.close()
                logger.info("[VoiceWorker] BullMQ 워커 정상 종료")
            except Exception as e:
                logger.warning(f"[VoiceWorker] 워커 종료 중 예외: {e}")
            finally:
                self._is_running = False
