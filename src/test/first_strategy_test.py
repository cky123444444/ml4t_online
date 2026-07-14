import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from src.ops.strategy.first_strategy import FirstStrategy

class TestFirstStrategy(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        os.environ['SCORE_DB_PATH'] = os.path.join(self._tmp_dir.name, 'scores.db')
        os.environ['ORDER_DB_PATH'] = os.path.join(self._tmp_dir.name, 'orders.db')

        # 初始化 FirstStrategy 实例
        self.model_preds = {
            'cur_price': 50001,
            'trmt_return_1h_out': [[10.0]],
            'ctrl_return_1h_out': [[13.0]],
            'trmt_sigma_1h_out': [[8.0]],
            'ctrl_sigma_1h_out': [[7.0]],
            't_out': [[0.45]],
        }
        self.strategy = FirstStrategy(self.model_preds)

    def tearDown(self):
        self._tmp_dir.cleanup()


    def test_execute_trading_logic(self):
        # Step 1: 执行 config_retriever
        print("\n[Step 1] Executing config_retriever...")
        self.strategy.config_retriever()

        # Step 2: 执行 value_tree
        print("\n[Step 2] Executing cal_value_tree...")
        self.strategy.cal_value_tree()
        print(f"Final Score: {self.strategy.final_score}")


        # Step 3: 执行 retrive_quantile
        print("\n[Step 3] Executing quantile_retriever...")
        self.strategy.quantile_retriever()
        print(f"Quantile: {self.strategy.quantile}")

        # Step 4: 执行 leverage_calc
        print("\n[Step 4] Executing cal_direction...")
        self.strategy.cal_direction()
        print(f"Direction: {self.strategy.direction}")

        # Step 2: 执行 value_tree
        print("\n[Step 5] Executing cal_leverage...")
        self.strategy.cal_leverage()
        print(f"Leverage: {self.strategy.leverage}")

        # Step 6: 执行 leverage_calc
        print("\n[Step 6] Executing cal_stop_loss_price...")
        self.strategy.cal_stop_loss_price()
        print(f"Stop Loss Price: {self.strategy.stop_loss_price}")



        # Step 6: 执行 leverage_calc
        print("\n[Step 7] Executing cal_position_rate...")
        self.strategy.cal_position_rate()
        print(f"Position Rate: {self.strategy.kelly_ratio}")

        # Step 6: 执行 is_valid
        print("\n[Step 8] Executing order_validate...")
        self.strategy.order_validate()
        print(f"Is Valid: {self.strategy.is_valid}")
        print(f"Fail Reason: {self.strategy.fail_reason}")
        # Step 6: 执行 is_valid
        print("\n[Step 9] Executing order_poster...")
        self.strategy.order_poster()
        print(f"Order Details: "
              f"final_score={self.strategy.final_score:.4f}, "
              f"quantile={self.strategy.quantile:.4f}, "
              f"direction={self.strategy.direction}, "
              f"leverage={self.strategy.leverage}, "
              f"cur_price={self.strategy.cur_price:.4f}, "
              f"stop_loss_price={self.strategy.stop_loss_price:.4f}, "
              f"position_ratio={self.strategy.kelly_ratio:.4f}, "
              f"is_valid={self.strategy.is_valid}, "
              f"fail_reason={self.strategy.fail_reason}")
        



if __name__ == '__main__':
    unittest.main()
