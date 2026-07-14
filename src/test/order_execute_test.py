import unittest
import os
import time
import tempfile
import shutil
from datetime import datetime, timezone

from src.ops.clients.binance_futures_client import BinanceFuturesClient
from src.ops.executor.account_router import AccountRouter
from src.ops.executor.order_placer import OrderPlacer
from src.ops.executor.order_closer import OrderCloser
from src.storage.order_repo import OrderRepo
from src.config.constants import SIDE_BUY, SIDE_SELL
from src.utils.logger import setup_logger
from src.ops.clients.client_registry import ClientRegistry
from src.ops.executor.order_finalizer import OrderFinalizer

logger = setup_logger("binance_futures_flow_test")

# ------------------ Test parameters ------------------
RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"

if RUN_INTEGRATION_TESTS:
    ClientRegistry.init_from_config(config_path="src/config/accounts.yaml", testnet=True)

class TestBinanceFuturesFlow(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with temporary database directory."""
        # Create a temporary directory for each test
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "order_repo.db")
        
        # Set environment variable for OrderRepo to use
        os.environ['ORDER_DB_PATH'] = self.db_path
        
        logger.info(f"Test database path: {self.db_path}")
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up environment variable
        if 'ORDER_DB_PATH' in os.environ:
            del os.environ['ORDER_DB_PATH']
        
        # Clean up temp files
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up test directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    @unittest.skipUnless(RUN_INTEGRATION_TESTS, "Skipping integration tests")
    def test_minimal_flow(self):
        """Minimal end-to-end flow: create order, assign, place, close"""
        # 1. Initialize account(s)

        # 2. Initialize repo, router, placer, closer
        repo = OrderRepo()
        router = AccountRouter()
        placer = OrderPlacer()
        closer = OrderCloser()
        finalizer = OrderFinalizer(max_retry=10)

        # 3. Step 1: Create NEW orders (simulate strategy)
        logger.info("Creating NEW orders")
        repo.create_order(symbol="BTCUSDT", direction="LONG", kelly_rate=0.1, leverage=1, stop_loss=30000)
        repo.create_order(symbol="BTCUSDT", direction="LONG", kelly_rate=0.01, leverage=1, stop_loss=30000)
        repo.create_order(symbol="BTCUSDT", direction="LONG", kelly_rate=0.02, leverage=1, stop_loss=30000)
        repo.create_order(symbol="BTCUSDT", direction="LONG", kelly_rate=0.03, leverage=1, stop_loss=30000)
        repo.create_order(symbol="BTCUSDT", direction="SHORT", kelly_rate=0.04, leverage=1, stop_loss=90000)
        repo.create_order(symbol="BTCUSDT", direction="SHORT", kelly_rate=0.05, leverage=1, stop_loss=90000)
        repo.create_order(symbol="BTCUSDT", direction="SHORT", kelly_rate=0.01, leverage=1, stop_loss=90000)
        repo.create_order(symbol="BTCUSDT", direction="SHORT", kelly_rate=0.03, leverage=1, stop_loss=90000)
        created_orders = repo.conn.execute("SELECT * FROM orders WHERE status='NEW'").fetchall()
        for order in created_orders:
            logger.info(f"Created order: {dict(order)}")

        # 4. Step 2: Route NEW orders to accounts
        logger.info("Routing NEW orders")
        router.route()

        # 5. Step 3: Place ASSIGNED orders
        assigned_orders = repo.conn.execute("SELECT * FROM orders WHERE status='ASSIGNED'").fetchall()
        for order in assigned_orders:
            logger.info(f"Assigned order: {order}")
        placer.place_orders()

        # 6. Step 4: Simulate holding period (optional sleep to test time-based close)
        logger.info("Sleeping 15 seconds to simulate holding")
        time.sleep(15)

        # 7. Step 5: Close OPEN orders if conditions met
        logger.info("Processing open orders for closure")
        closer.process_open_orders()

        # 8. Step 6: Finalize and show final PnL
        finalizer.finalize_orders()
        logger.info("Orders finalized")


if __name__ == "__main__":
    unittest.main()
