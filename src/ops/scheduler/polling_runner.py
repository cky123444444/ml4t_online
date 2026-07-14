import time
import uuid
from src.ops.executor.account_router import AccountRouter
from src.ops.executor.order_placer import OrderPlacer
from src.ops.executor.order_closer import OrderCloser
from src.ops.executor.order_finalizer import OrderFinalizer
from src.ops.clients.client_registry import ClientRegistry
from src.utils.logger import setup_logger, set_request_id

logger = setup_logger("polling_runner")
ClientRegistry.init_from_config(config_path="src/config/accounts.yaml", testnet=True)

class PollingRunner:
    def __init__(self):
        self.router = AccountRouter()
        self.placer = OrderPlacer()
        self.closer = OrderCloser()
        self.finalizer = OrderFinalizer()

    def run_once(self):
        # 为每个轮询周期生成新的 request_id
        set_request_id(f"poll-{uuid.uuid4().hex[:8]}")

        self.router.route()
        self.placer.place_orders()
        self.closer.process_open_orders()
        self.finalizer.finalize_orders()

    def run_forever(self, interval_sec: int = 60):
        logger.info(f"Polling runner started, interval={interval_sec}s")
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.exception(f"Polling cycle failed: {e}")
            finally:
                time.sleep(interval_sec)