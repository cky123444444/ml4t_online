import threading
from src.ops.scheduler.polling_runner import PollingRunner
from src.utils.logger import setup_logger

logger = setup_logger("background_polling")


def start_polling_thread(interval_sec: int = 60):
    """
    Start polling runner in a daemon thread.
    """
    runner = PollingRunner()

    thread = threading.Thread(
        target=runner.run_forever,
        kwargs={"interval_sec": interval_sec},
        name="PollingRunnerThread",
        daemon=True,   # Daemon thread will exit when main program exits
    )

    thread.start()
    logger.info("Polling runner thread started")

    return thread
