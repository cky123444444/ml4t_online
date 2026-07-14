@staticmethod
def calc_avg_price(trades: list[dict]) -> float:
    total_qty = 0.0
    total_notional = 0.0

    for t in trades:
        qty = abs(float(t["qty"]))
        price = float(t["price"])
        total_qty += qty
        total_notional += qty * price

    return total_notional / total_qty if total_qty > 0 else 0.0

@staticmethod
def calc_commission(trades: list[dict]) -> float:
    return sum(float(t.get("commission", 0.0)) for t in trades)

