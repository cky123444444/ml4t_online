from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from src.utils.logger import setup_logger
from src.config import constants

logger = setup_logger('binance_futures_client')

class BinanceFuturesClient:
    def __init__(self, api_key, api_secret, symbol=constants.BTCUSDT_SYMBOL, testnet=True):
        """
        Initialize Binance Futures Client
        :param api_key: Binance API key
        :param api_secret: Binance API secret
        :param symbol: futures trading pair, fixed for this instance
        :param testnet: use testnet or real
        """
        self.logger = logger
        self._taker_fee_rate = 0.0005  # default 0.05%
        self.symbol = symbol
        self.client = Client(api_key, api_secret, testnet=testnet)
        # self.client.futures_change_position_mode(dualSidePosition=True)
        if testnet:
            self.client.FUTURES_URL = constants.BINANCE_TESTNET_URL

        # ------------------ Hedge Mode (default) ------------------
        try:
            # check for any open positions
            positions = self.client.futures_position_information()
            has_position = any(float(p['positionAmt']) != 0 for p in positions)
            if has_position:
                self.logger.warning("Cannot enable hedge mode: there are open positions")
            else:
                self.client.futures_change_position_mode(dualSidePosition=True)
                self.logger.info("Hedge mode enabled by default")
        except Exception as e:
            self.logger.error(f"Failed to enable hedge mode: {e}")

    # ------------------ Taker Fee ------------------
    @property
    def taker_fee_rate(self):
        return self._taker_fee_rate

    @taker_fee_rate.setter
    def taker_fee_rate(self, rate):
        self._taker_fee_rate = rate

    def get_mark_price(self) -> float:
        info = self.client.futures_mark_price(symbol=self.symbol)
        # self.logger.info(f"Market price for {self.symbol}: {info.get('markPrice', 0.0)}")
        return float(info.get('markPrice', 0.0))
    
    # ------------------ Positions --------------
    def get_total_position_notional(self) -> float:
        """
        Calculate total notional value of all open positions (without leverage).
        """
        try:
            total_notional = 0.0
            positions = self.client.futures_position_information()

            for p in positions:
                amt = float(p.get("positionAmt", 0.0))
                if amt == 0:
                    continue

                mark_price = float(p.get("markPrice", 0.0))
                total_notional += abs(amt) * mark_price

            self.logger.info(
                f"Total position notional (no leverage): {total_notional}"
            )
            return total_notional

        except Exception as e:
            self.logger.warning(f"Failed to calculate total position notional: {e}")
            return 0.0

    
    # ------------------ Order ------------------
    def get_order(self, order_id: str) -> dict:
        """
        Fetch a futures order by order_id.
        """
        try:
            order = self.client.futures_account_trades(
                symbol=self.symbol,
                orderId=order_id
            )
            return order
        except Exception as e:
            self.logger.error(f"Failed to fetch order {order_id}: {e}")
            return {}
        
    def get_order_trades(self, order_id: str) -> list[dict]:
        """
        Fetch all trades for a given futures order.
        """
        try:
            trades = self.client.futures_account_trades(
                symbol=self.symbol,
                orderId=order_id,
            )
            return trades or []
        except Exception as e:
            self.logger.error(f"Failed to fetch trades for order {order_id}: {e}")
            return []


    # ------------------ Kline ------------------
    def get_klines(self, interval='1h', limit=100):
        """Fetch kline/candlestick data"""
        try:
            return self.client.futures_klines(symbol=self.symbol, interval=interval, limit=limit)
        except (BinanceAPIException, BinanceRequestException) as e:
            self.logger.error(f"Failed to get klines for {self.symbol}: {e}")
            return []

    # ------------------ Order ------------------
    def place_order(self, side=constants.SIDE_BUY, quantity=0, leverage=None, order_type='MARKET', position_side='BOTH'):
        """Place a futures order with optional leverage and position side"""
        try:
            # set leverage if provided
            if leverage is not None:
                self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)

            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                positionSide=position_side
            )
            # self.logger.info(f"Placed order: {order}")
            return order

        except (BinanceAPIException, BinanceRequestException) as e:
            self.logger.error(f"Failed to place order for {self.symbol}: {e}")
            return None

    # ------------------ Close Position ------------------
    def close_position(self):
        """Close all positions for the symbol"""
        responses = []
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for p in positions:
                amt = float(p['positionAmt'])
                if amt == 0:
                    continue
                side = constants.SIDE_SELL if amt > 0 else constants.SIDE_BUY
                quantity = abs(amt)
                pos_side = p['positionSide']
                resp = self.place_order(side, quantity, order_type='MARKET', position_side=pos_side)
                if resp:
                    self.logger.info(f"Closed {pos_side} position: {quantity} {self.symbol}")
                responses.append(resp)
        except Exception as e:
            self.logger.error(f"Failed to close positions for {self.symbol}: {e}")
        return responses

    # ------------------ PnL ------------------
    @property
    def realized_pnl(self):
        """Fetch realized PnL (already closed positions) for LONG and SHORT separately"""
        pnl = {"LONG": 0.0, "SHORT": 0.0}
        try:
            trades = self.client.futures_account_trades(symbol=self.symbol)
            for t in trades:
                pos_side = t.get("positionSide", "BOTH")  # LONG / SHORT
                if pos_side in pnl:
                    pnl[pos_side] += float(t['realizedPnl']) - float(t['commission'])
        except Exception as e:
            self.logger.error(f"Failed to fetch realized PnL: {e}")
        return pnl
    
    @property
    def unrealized_pnl(self):
        """Estimate unrealized PnL for open positions, separately for LONG and SHORT"""
        pnl = {"LONG": 0.0, "SHORT": 0.0}
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for p in positions:
                amt = float(p['positionAmt'])
                if amt == 0:
                    continue
                pos_side = p.get("positionSide", "BOTH")
                entry = float(p['entryPrice'])
                mark = float(p['markPrice'])
                pnl_value = (mark - entry) * amt
                pnl_value -= abs(amt) * mark * self.taker_fee_rate
                if pos_side in pnl:
                    pnl[pos_side] += pnl_value
        except Exception as e:
            self.logger.error(f"Failed to fetch unrealized PnL: {e}")
        return pnl
    
    @property
    def total_estimated_pnl(self):
        """Sum of realized + unrealized PnL, separately for LONG and SHORT"""
        realized = self.realized_pnl
        unrealized = self.unrealized_pnl
        total = {"LONG": realized.get("LONG", 0.0) + unrealized.get("LONG", 0.0),
             "SHORT": realized.get("SHORT", 0.0) + unrealized.get("SHORT", 0.0)}
        return total
    
    @property
    def current_position(self) -> dict:
        """Return net positions for LONG and SHORT"""
        result = {"LONG": 0.0, "SHORT": 0.0}
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for p in positions:
                side = p.get("positionSide")
                if side in result:
                    result[side] = float(p.get("positionAmt", 0.0))
        except Exception as e:
            self.logger.warning(f"Failed to fetch current positions: {e}")

        return result

    @property
    def residual_position(self) -> dict:
        """Return residual positions for LONG and SHORT (absolute)"""
        result = {"LONG": 0.0, "SHORT": 0.0}

        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for p in positions:
                side = p.get("positionSide")
                if side in result:
                    result[side] = abs(float(p.get("positionAmt", 0.0)))
        except Exception as e:
            self.logger.warning(f"Failed to fetch residual positions: {e}")

        return result
    
    @property
    def available_balance(self) -> float:
        try:
            info = self.client.futures_account()
            for asset in info.get("assets", []):
                if asset["asset"] == "USDT":
                    return float(asset["availableBalance"])
            return 0.0
        except Exception as e:
            self.logger.warning(f"Failed to fetch available balance: {e}")
            return 0.0
        
    def get_total_asset(self) -> float:
        """
        Total asset = available_balance + total position notional (no leverage)
        """
        try:
            available = self.available_balance
            position_notional = self.get_total_position_notional()
            total_asset = available + position_notional

            self.logger.info(
                f"Total asset calculated: "
                f"available={available}, "
                f"position_notional={position_notional}, "
                f"total={total_asset}"
            )
            return total_asset

        except Exception as e:
            self.logger.error(f"Failed to calculate total asset: {e}")
            return 0.0
        
    def get_total_asset_btc(self) -> float:
        """
        Total asset in BTC:
        (available_balance + position_notional) / mark_price
        """
        try:
            total_usdt = self.get_total_asset()
            mark_price = self.get_mark_price()

            if mark_price <= 0:
                return 0.0

            total_btc = total_usdt / mark_price

            self.logger.info(
                f"Total asset (BTC): "
                f"usdt={total_usdt}, "
                f"mark_price={mark_price}, "
                f"btc={total_btc}"
            )
            return total_btc

        except Exception as e:
            self.logger.warning(f"Failed to calculate total asset in BTC: {e}")
            return 0.0