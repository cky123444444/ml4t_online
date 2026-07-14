import json
import os
from datetime import datetime, timezone
import numpy as np
from sre_constants import ASSERT
import time

from .base_strategy import BaseStrategy
from src.utils.logger import setup_logger
from src.storage.score_repo import ScoreRepo
from src.storage.order_repo import OrderRepo

logger = setup_logger('ple_strategy')
CONFIG_FILE = 'strategy_config.json'

class PleStrategy(BaseStrategy):
    def __init__(self, model_preds):
        # Variables
        self.direction = ''  # 1 for long, 0 for short
        self.leverage = 0.0     # Leverage multiplier, default is 0
        self.stop_loss_price = 0.0  # Stop-loss price
        # Ensure all inputs are converted to native Python floats (CPU-compatible)
        self.cur_price = float(model_preds['cur_price'])  # 强制转换为 float
        self.target_price = 0.0  # Target price
        self.kelly_ratio = 0.0
        self.is_valid = True
        self.fail_reason = ''

        # Private variables - all converted to float
        """
            "ctrl_return_1h_out",
            "trmt_return_1h_out",
            "ctrl_sigma_1h_out",
            "trmt_sigma_1h_out",
            "t_out",
            "return_1h_eps_out",
            "sigma_1h_eps_out" 到 输入的映射关系
        """
        self.y0_trmt = np.clip(np.expm1(float(model_preds['trmt_return_1h_out'][0]))/10.0, 0.0, 30.0)
        self.y0_ctrl = np.clip(np.expm1(float(model_preds['ctrl_return_1h_out'][0]))/10.0, 0.0, 30.0)
        self.y1_trmt = np.clip(np.expm1(float(model_preds['trmt_sigma_1h_out'][0]))/10.0, 0.0, 25.0)
        self.y1_ctrl = np.clip(np.expm1(float(model_preds['ctrl_sigma_1h_out'][0]))/10.0, 0.0, 25.0)
        self.t_ctrl = float(model_preds['t_ctrl_out'][0])
        self.t_trmt = float(model_preds['t_trmt_out'][0])
        self.t = np.exp(self.t_trmt) / (np.exp(self.t_trmt) + np.exp(self.t_ctrl))
        
        self.final_score = 0.0
        self.quantile = -999.0
        self.score_repo = ScoreRepo()
        self.order_repo = OrderRepo()

        # self.print_info()

    def print_info(self):
        logger.info(f"PleStrategy Info: y0_trmt={self.y0_trmt}, y0_ctrl={self.y0_ctrl}, "
                    f"y1_trmt={self.y1_trmt}, y1_ctrl={self.y1_ctrl}, t_ctrl={self.t_ctrl}, t_trmt={self.t_trmt}, "
                    f"cur_price={self.cur_price}")


    def execute(self):
        return self.execute_trading_logic()

    def execute_trading_logic(self):
        """
        Execute trading logic based on quantile thresholds.
        """
        # Load configuration and initialize Redis client
        if not self.config_retriever():
            logger.warning("Configuration retrieval failed. Using default settings.")

        if self.y0_trmt is None or self.y0_ctrl is None or self.t is None:
            logger.warning(f"y0_trmt: {self.y0_trmt}, y0_ctrl: {self.y0_ctrl}, t: {self.t}. Skipped final score calculation.")
            self.is_valid = False
            if not self.fail_reason:
                self.fail_reason = '[fail] Missing values for cal_value_tree calculation'
        else:
            self.cal_value_tree()
            # 查看当前分数在队列中的分位数,拿到direction
            self.quantile_retriever()
            self.cal_direction(mode=self.quantile_mode)
            self.cal_leverage()
            self.cal_stop_loss_price()
            self.cal_position_rate()
            # Calculate stop-loss price after validation
            # only read permission to db, write operation is in binance_client
            self.order_validate()
        self.order_poster()
        if not self.fail_reason:
            logger.info("ple_strategy completed successfully.")
        else:
            logger.info(self.fail_reason)
    
    def config_retriever(self):
        """
        Load configuration from the JSON file and set instance variables.
        Initialize Redis client.
        """
        config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.info(f"Error opening or parsing config file: {e}, using default settings.")
        else:
            logger.info(f"Configuration file not found at {config_path}, using default settings.")
            

        # Queue configuration
        self.score_key = config.get('queue_config', {}).get('score_key', 'quantile_scores')
        self.top_threshold = config.get('queue_config', {}).get('top_threshold', 0.9)
        self.bottom_threshold = config.get('queue_config', {}).get('bottom_threshold', 0.1)
        self.queue_length = config.get('queue_config', {}).get('queue_length', 1044)
        self.k = config.get('queue_config', {}).get('k', 1.0)
        self.quantile_mode = config.get('queue_config', {}).get('quantile_mode', 2)
        
        # Execution configuration
        self.min_position = config.get('execute_config', {}).get('min_position', 0.001)
        self.min_long_threshold = config.get('execute_config', {}).get('min_long_threshold', 10)
        self.min_short_threshold = config.get('execute_config', {}).get('min_short_threshold', 10)
        self.kelly_factor = config.get('execute_config', {}).get('kelly_factor', 0.5)
        self.sigma_mul = config.get('execute_config', {}).get('sigma_mul', 3)

        # Leverage configuration
        leverage_config = config.get('leverage_config', {})
        self.sharpe_bins = leverage_config.get('sharpe_ratio_bins', [0, 1, 1.2, 1.5, 2])
        self.leverage_list = leverage_config.get('leverage_list', [0, 2, 3, 5, 10])

        return True
    
    def cal_value_tree(self):
        # calculate final score 
        up_score = np.exp(self.t_trmt) / (np.exp(self.t_trmt) + np.exp(self.t_ctrl))
        down_score = np.exp(self.t_ctrl) / (np.exp(self.t_trmt) + np.exp(self.t_ctrl))
        self.final_score = up_score
        logger.info(f"Calculated final_score: {self.final_score} (up_score: {up_score}, down_score: {down_score})")
        return True
    
    
    
    def quantile_retriever(self):
        try:
            # 从db获取当前final_score的分位数
            self.quantile, num_rows = self.score_repo.fetch_quantile(self.final_score, self.queue_length)
            if self.quantile == -999.0:
                logger.warning(f"Insufficient data for quantile calculation, window_size: {self.queue_length} current rows: {num_rows}.")
                self.is_valid = False
                if not self.fail_reason:
                    self.fail_reason = '[fail] Insufficient data for quantile_retriever'
                return False
            logger.info(f"Retrieved quantile: {self.quantile} for final_score: {self.final_score} (cur_rows: {num_rows})")
        except Exception as e:
            logger.error(f"Error retrieving or processing quantile: {e}")
            self.quantile = -999.0
            if not self.fail_reason:
                self.fail_reason = '[fail] quantile_retriever failed'
            self.is_valid = False
            return False
        return True



    def cal_direction(self, mode=1):
        if self.quantile == -999.0:
            logger.info("Quantile not found or invalid.")
            if not self.fail_reason:
                self.fail_reason = '[fail] quantile is None in cal_direction'
            self.is_valid = False
            return False
        # dynamic quantile threshold adjustment
        if mode == 2:
            num_order_60_long = self.score_repo.count_recent_by_direction('LONG', window_size=60)
            num_order_60_short = self.score_repo.count_recent_by_direction('SHORT', window_size=60)
            logger.info(f"Recent 60min orders - LONG: {num_order_60_long}, SHORT: {num_order_60_short}")
            self.top_threshold = 1.0 - float((1.0 - self.top_threshold) / (self.k * num_order_60_long + 1))
            self.bottom_threshold = float(self.bottom_threshold / (self.k * num_order_60_short + 1))

        if self.quantile >= self.top_threshold:
            self.direction = 'LONG'  # Set direction to long
            logger.info(f"Quantile {self.quantile} is above the upper threshold {self.top_threshold}. Valid for long.")
            
        elif self.quantile <= self.bottom_threshold:
            self.direction = 'SHORT'  # Set direction to short
            logger.info(f"Quantile {self.quantile} is below the lower threshold {self.bottom_threshold}. Valid for short.")

        else:
            logger.warning(f"Quantile {self.quantile} is not within the trading thresholds upper_threshold {self.top_threshold}, lower_threshold {self.bottom_threshold}. Skipped.")
            if not self.fail_reason:
                self.fail_reason = '[fail] quantile not in trading thresholds in cal_direction'
            self.is_valid = False
            return False
        return True
    
    def cal_leverage(self):
        if self.direction == '':
            logger.info("Direction is not set. Cannot calculate leverage.")
            if not self.fail_reason:
                self.fail_reason = '[fail] direction is empty in cal_leverage'
            self.is_valid = False
            return False
        # elif self.direction == 'LONG':
        #     volatility = max(self.y1_trmt - self.y0_trmt/2.0 ,0)*self.t + (self.y1_ctrl + self.y0_ctrl/2.0 - max(self.y1_ctrl - self.y0_ctrl/2.0 ,0))* (1 - self.t) + 1e-6
        # elif self.direction == 'SHORT':
        #     volatility = max(self.y1_ctrl - self.y0_ctrl/2.0 ,0) * (1-self.t) + (self.y1_trmt + self.y0_trmt/2.0 - max(self.y1_trmt - self.y0_trmt/2.0 ,0)) * self.t + 1e-6
        
        # sharpe_ratio = abs(self.final_score) / volatility
        # print(f"cky calculated sharpe_ratio: {sharpe_ratio}")
        # if sharpe_ratio < 0:
        #     logger.info(f"Calculated Sharpe ratio {sharpe_ratio} is negative. Cannot calculate leverage.")
        #     if not self.fail_reason:
        #         self.fail_reason = '[fail] negative sharpe_ratio in cal_leverage'
        #     self.is_valid = False
        #     return False
        # # Determine leverage based on Sharpe ratio bins
        # for i in range(len(self.sharpe_bins) - 1):
        #     if self.sharpe_bins[i] <= sharpe_ratio < self.sharpe_bins[i + 1]:
        #         self.leverage = self.leverage_list[i]
        #         logger.info(f"Sharpe ratio {sharpe_ratio} falls into bin [{self.sharpe_bins[i]}, {self.sharpe_bins[i + 1]}). "
        #               f"Leverage set to {self.leverage}.")
        #         return True

        # If Sharpe ratio exceeds the last bin, use the last leverage value
        self.leverage = self.leverage_list[-1]
        logger.info(f"Leverage set to {self.leverage}.")
        return True
    
    def cal_stop_loss_price(self):
        """
        Calculate the stop-loss price based on the current price and y1_trmt, y1_ctrl.
        """
        if self.cur_price is None or self.y1_trmt is None or self.y1_ctrl is None or self.direction == '':
            logger.info("Current info is not set. Cannot calculate stop-loss price.")
            if not self.fail_reason:
                self.fail_reason = '[fail] Missing values in cal_stop_loss_price'
            self.is_valid = False
            return False
        
        if self.direction == '':
            logger.info("Direction is not set. Cannot calculate stop-loss price.")
            if not self.fail_reason:
                self.fail_reason = '[fail] direction is empty in cal_stop_loss_price'
            self.is_valid = False
            return False
        
        # Example calculation for stop-loss price
        if self.direction == 'LONG':  # Long position
            self.target_price = self.cur_price * (1 + self.y0_trmt / 1000)
            mid_price = (self.target_price + self.cur_price) / 2
            self.stop_loss_price = mid_price - self.sigma_mul * mid_price* self.y1_trmt/1000
            while self.stop_loss_price > self.cur_price - mid_price* self.y1_trmt/1000:
                logger.info("Adjusting stop-loss price for long position.")
                self.stop_loss_price -= mid_price * self.y1_trmt/1000

        elif self.direction == 'SHORT':  # Short position
            self.target_price = self.cur_price * (1 - self.y0_ctrl / 1000)
            mid_price = (self.target_price + self.cur_price) / 2
            self.stop_loss_price = mid_price + self.sigma_mul * mid_price * self.y1_ctrl/1000
            while self.stop_loss_price < self.cur_price + mid_price * self.y1_ctrl/1000:
                logger.info("Adjusting stop-loss price for short position.")
                self.stop_loss_price += mid_price * self.y1_ctrl/1000

        logger.info(f"Stop-loss price calculated: {self.stop_loss_price}")
        return True
    
    def cal_position_rate(self):
        if self.stop_loss_price == None:
            logger.info("Stop-loss price is not set. Cannot calculate position rate.")
            if not self.fail_reason:
                self.fail_reason = '[fail] stop_loss_price is empty in cal_position_rate'
            self.is_valid = False
            return False
        # Magic number need to be tuned
        # if self.direction == 'LONG':
        #     l_win = self.target_price - self.cur_price - self.cur_price * 0.001  # 1bp trading fee
        #     # On condition stop_loss_price should be smaller than target price
        #     l_loss = self.cur_price - self.stop_loss_price + self.cur_price * 0.001  # 1bp trading fee
        #     position_rate = (self.t - (1 - self.t) / (l_win / l_loss + 1e-6)) * self.kelly_factor  # Half Kelly
        #     logger.info(f"half kelly position rate: {position_rate}")
        #     self.kelly_ratio = position_rate
            
        # elif self.direction == 'SHORT':
        #     l_win = self.cur_price - self.target_price - self.cur_price * 0.001  # 1bp trading fee
        #     # On condition stop_loss_price should be smaller than target price
        #     l_loss = self.stop_loss_price - self.cur_price + self.cur_price * 0.001  # 1bp trading fee
        #     position_rate = ((1 - self.t) - self.t / (l_win / l_loss + 1e-6)) * self.kelly_factor  # Half Kelly
        #     self.kelly_ratio = position_rate
        self.kelly_ratio = 0.1  # Fixed position rate for simplicity
        return True
           
    def order_validate(self):
        """
        Check if the trading conditions are valid based on quantile thresholds.
        
        :param min_position: The minimum residual position required to proceed with trading.
        :return: True if trading conditions are met, False otherwise.
        """
        if self.leverage == 0.0:
            logger.warning("Leverage is set to 0. Skipping trade execution.")
            if not self.fail_reason:
                self.fail_reason = '[fail] leverage is 0 in order_validate'
            self.is_valid = False
            return False
        
        if self.kelly_ratio <= 0.0:
            logger.warning(f"Kelly ratio {self.kelly_ratio} is non-positive. Skipping trade execution.")
            if not self.fail_reason:
                self.fail_reason = '[fail] kelly_ratio <= 0 in order_validate'
            self.is_valid = False
            return False
        
        if self.y0_trmt < self.min_long_threshold and self.direction == 'LONG':
                logger.info(f"y0_trmt {self.y0_trmt} is below the threshold {self.min_long_threshold}.")
                if not self.fail_reason:
                    self.fail_reason = '[fail] y0_trmt below min_long_threshold in order_validate'
                self.is_valid = False

        if self.y0_ctrl < self.min_short_threshold and self.direction == 'SHORT':
                logger.info(f"y0_ctrl {self.y0_ctrl} is below the threshold {self.min_short_threshold}.")
                if not self.fail_reason:
                    self.fail_reason = '[fail] y0_ctrl below min_short_threshold in order_validate'
                self.is_valid = False

        ########################################
        #TBD
        # rule-based strategy
        ########################################
        return True
        




    def order_poster(self):
        """
        Create a NEW order in the database for AccountRouter to process.
        The order contains direction, quantity (kelly_ratio), leverage, and stop-loss price.

        :return: True if order created successfully, False otherwise.
        """
        if self.is_valid:
             # Create order in order_repo with status=NEW for AccountRouter to pick up
            try: 
                self.order_repo.create_order(
                    symbol='BTCUSDT',
                    direction=self.direction,
                    kelly_rate=self.kelly_ratio,  # position ratio as kelly_rate
                    leverage=int(self.leverage),
                    stop_loss=self.stop_loss_price,
                )

                logger.info(f"Successfully created NEW order: symbol=BTCUSDT, direction={self.direction}, "
                        f"kelly_rate={self.kelly_ratio}, leverage={self.leverage}, stop_loss={self.stop_loss_price}")
            except Exception as e:
                if not self.fail_reason:
                    self.fail_reason = f'[fail] order_repo create_order failed: {e}'
                logger.error(f"Error creating order: {e}")
             
        try:
            # Record score for historical analysis
            self.score_repo.insert_score(
                timestamp=int(datetime.now(timezone.utc).timestamp()),
                final_score=self.final_score,
                quantile=self.quantile,
                direction=self.direction,
                leverage=self.leverage,
                cur_price=self.cur_price,
                stop_loss_price=self.stop_loss_price,
                position_ratio=self.kelly_ratio,
                is_valid=self.is_valid,
                fail_reason=self.fail_reason
            )

            return True
        except Exception as e:
            if not self.fail_reason:
                self.fail_reason = f'[fail] score_repo insert_score failed: {e}'
            logger.error(f"Error creating order: {e}")
            return False