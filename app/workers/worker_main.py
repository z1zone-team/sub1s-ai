import asyncio
import logging
import signal
import sys

from app.configs.settings import settings
from app.pipelines.stt_pipeline import stt_pipeline
from app.workers.voice_worker import VoiceWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [BullMQ-Worker] %(name)s: %(message)s",
)
logger = logging.getLogger("worker_main")


async def preload_model() -> bool:
    """
    워커를 시작하기 전에 Whisper 모델을 로드한다. 단독 실행(`worker_main`)에서는 FastAPI의 시작 단계가 없어
    모델이 첫 작업 때 로드되고, 그 작업만 로드 시간(수십 초)만큼 늦어 서버의 STT 대기 제한(25초)에 걸릴 수 있다.
    로드에 실패해도 워커는 뜬다(작업은 `음성 인식 모델을 사용할 수 없습니다`로 실패하고 다음 작업에서 다시 시도한다)
    """
    if not settings.whisper_preload:
        return False
    try:
        # 모델 로드는 동기이고 오래 걸려서 이벤트 루프를 막지 않도록 스레드에서 한다
        await asyncio.to_thread(stt_pipeline.load_model)
        return True
    except Exception as e:
        logger.warning(f"Whisper 모델을 미리 로드하지 못했습니다. 첫 작업에서 다시 시도합니다: {e}")
        return False


async def run_worker():
    logger.info("Sub1S AI BullMQ Voice Worker 프로세스를 시작합니다...")
    logger.info(
        f"대상 Valkey: {settings.valkey_host}:{settings.valkey_port}, 큐: '{settings.voice_queue_name}'"
    )

    await preload_model()

    worker = VoiceWorker()
    stop_event = asyncio.Event()

    # 종료 시그널 핸들러
    def handle_signal():
        logger.info("종료 시그널 수신, 워커 종료 절차를 시작합니다...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            # Windows에서는 loop.add_signal_handler가 미지원될 수 있음
            pass

    started = await worker.start()
    if not started:
        logger.warning(
            "Valkey에 즉시 연결되지 않았습니다. 백그라운드 재연결 루프를 실행합니다."
        )

    # 주기적 재연결 및 대기 루프
    try:
        while not stop_event.is_set():
            if not worker._is_running:
                available = await worker.is_valkey_available()
                if available:
                    logger.info("Valkey 연결 감지, 워커를 활성화합니다.")
                    await worker.start()
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass
    finally:
        await worker.stop()
        logger.info("BullMQ Worker 프로세스가 안전하게 종료되었습니다.")


if __name__ == "__main__":
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("프로세스 인터럽트 종료")
        sys.exit(0)
