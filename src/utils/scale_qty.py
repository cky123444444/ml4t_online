import math

def scale_quantity(qty: float, precision: int = 3) -> float:
    """
    Binance BTCUSDT quantity:
    - keep `precision` decimals
    - always round DOWN
    """
    factor = 10 ** precision
    return math.floor(qty * factor) / factor
