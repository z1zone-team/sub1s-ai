import asyncio
import logging
import signal
import sys

from app.configs.settings import settings
from app.workers.voice_worker import VoiceWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [BullMQ-Worker] %(name)s: %(message)s",
)
logger = logging.getLogger("worker_main")


async def run_worker():
    logger.info("Sub1S AI BullMQ Voice Worker 프로세스를 시작합니다...")
    logger.info(
        f"대상 Valkey: {settings.valkey_host}:{settings.valkey_port}, 큐: '{settings.voice_queue_name}'"
    )

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
