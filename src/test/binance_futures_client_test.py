import unittest
import os
from src.ops.clients.binance_futures_client import BinanceFuturesClient
from src.ops.clients.client_registry import ClientRegistry
from src.utils.logger import setup_logger
from src.config.constants import SIDE_BUY, SIDE_SELL

logger = setup_logger('binance_futures_client_test')
# ------------------ Test parameters ------------------
RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"


@unittest.skipUnless(RUN_INTEGRATION_TESTS, "Skipping integration tests")
class TestBinanceFuturesClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the Binance Futures Client once for all tests"""
        ClientRegistry.init_from_config(config_path="src/config/accounts.yaml", testnet=True)
        cls.client = ClientRegistry.get_client("Reinhardt")

    def test_get_klines(self):
        """Fetch latest klines"""
        klines = self.client.get_klines(limit=5)
        logger.info(f"Latest 5 klines: {klines}")
        self.assertIsInstance(klines, list)
    
    def test_open_long_and_short_then_close(self):
        """Open LONG and SHORT positions in hedge mode, then close both"""
        logger.info("Testing hedge mode: open LONG and SHORT")

        qty = 0.005

        # -------- open LONG --------
        order_long = self.client.place_order(
            side=SIDE_BUY,
            quantity=qty,
            position_side='LONG'
        )
        
        logger.info(f"LONG order response: {order_long}")
        self.assertTrue(order_long is None or "orderId" in order_long)

        # -------- open SHORT --------
        order_short = self.client.place_order(
            side=SIDE_SELL,
            quantity=qty,
            position_side='SHORT'
        )   
        logger.info(f"SHORT order response: {order_short}")
        self.assertTrue(order_short is None or "orderId" in order_short)

        # -------- verify positions --------
        positions = self.client.client.futures_position_information(
            symbol=self.client.symbol
        )

        long_amt = 0.0
        short_amt = 0.0

        for p in positions:
            amt = float(p["positionAmt"])
            if p["positionSide"] == "LONG":
                long_amt = amt
            elif p["positionSide"] == "SHORT":
                short_amt = amt

        logger.info(f"LONG position amount: {long_amt}")
        logger.info(f"SHORT position amount: {short_amt}")

        self.assertNotEqual(long_amt, 0.0, "LONG position should exist")
        self.assertNotEqual(short_amt, 0.0, "SHORT position should exist")
        # -------- close all --------
        close_response = self.client.close_position()
        logger.info(f"Close positions response: {close_response}")
        self.assertIsInstance(close_response, list)


    def test_close_position(self):
        """Close all positions"""
        close_response = self.client.close_position()
        logger.info(f"Close positions response: {close_response}")
        self.assertIsInstance(close_response, list)

    def test_pnl_properties(self):
        """Check PnL properties"""
        logger.info(f"Realized PnL: {self.client.realized_pnl}")
        logger.info(f"Unrealized PnL: {self.client.unrealized_pnl}")
        logger.info(f"Total Estimated PnL: {self.client.total_estimated_pnl}")

        self.assertIsInstance(self.client.realized_pnl, dict)
        self.assertIsNotNone(self.client.unrealized_pnl)
        self.assertIsInstance(self.client.total_estimated_pnl, dict)

if __name__ == "__main__":
    unittest.main()
