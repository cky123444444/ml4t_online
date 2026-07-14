# Testnet domain
BINANCE_TESTNET_URL = "https://testnet.binancefuture.com"


# ------------------ Trading Constants ------------------
# symbols
BTCUSDT_SYMBOL = "BTCUSDT"

# Default taker fee rate
DEFAULT_TAKER_FEE = 0.0005 # 0.05%
DEFAULT_MARGIN_SAFETY_FACTOR = 1.05  # 5% extra margin

# Buy / Sell sides
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

# Action mapping
SIDE_ACTION_OPEN_MAP= {
    "LONG": SIDE_BUY,
    "SHORT": SIDE_SELL
}
SIDE_ACTION_CLOSE_MAP= {
    "LONG": SIDE_SELL,
    "SHORT": SIDE_BUY
}

# Order status
ORDER_STATUS_NEW = "NEW"
ORDER_STATUS_ASSIGNED = "ASSIGNED"
ORDER_STATUS_OPEN = "OPEN"
ORDER_STATUS_CLOSED = "CLOSED"
ORDER_STATUS_FAILED = "FAILED"
ORDER_STATUS_SETTLED = "SETTLED"

# Order failed reasons
FAILED_REASON_MAX_RETRY = "Max attempts hit"
FAILED_REASON_NO_AVAILABLE_ACCOUNT = "No elegible account found"
FAILED_REASON_ASSIGNED_ACCOUNT_NOT_FOUND = "Assigned account not found"
