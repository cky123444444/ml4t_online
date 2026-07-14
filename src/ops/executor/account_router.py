import os
from src.storage.order_repo import OrderRepo
from src.ops.clients.client_registry import ClientRegistry
from src.utils.logger import setup_logger
from src.utils.scale_qty import scale_quantity
from src.config.constants import DEFAULT_MARGIN_SAFETY_FACTOR, ORDER_STATUS_FAILED, ORDER_STATUS_ASSIGNED, FAILED_REASON_NO_AVAILABLE_ACCOUNT

logger = setup_logger("account_router", log_file_name="executor.log")
MARGIN_SAFETY_FACTOR = os.getenv("MARGIN_SAFETY_FACTOR", DEFAULT_MARGIN_SAFETY_FACTOR)

class AccountRouter:
    """
    AccountRouter is responsible for assigning NEW orders to a suitable account.

    Responsibilities:
    - Fetch orders with status = NEW
    - Select an available account for each order
    - Update order status to ASSIGNED or FAILED
    """

    def __init__(self, max_retry: int = 5):
        """
        :param max_retry: max retry count before marking order as FAILED
        """
        self.repo = OrderRepo()
        self.max_retry = max_retry

    def route(self):
        """
        Route NEW orders to suitable accounts.

        Routing rules:
        - Account must have ZERO position on the required direction
        - Among eligible accounts, choose the one with the smallest residual position
        - If no account is eligible, mark order as FAILED
        """
        orders = self.repo.fetch_new_orders()
        if not orders:
            # logger.info("No NEW orders to route")
            return

         # fetch all active orders (ASSIGNED or OPEN) once, build a lookup
        active_orders = self.repo.fetch_active_orders()  # assume returns dict/list of dict
        active_map = {}  # key = (account_name, direction)
        for o in active_orders:
            account = o.get("assigned_account")
            direction = o.get("direction")
            if account and direction:
                active_map.setdefault((account, direction), []).append(o)

        for order in orders:
            try:
                self._route_single_order(order, active_map)
            except Exception as e:
                logger.exception(f"Unexpected error while routing order {order['id']}: {e}")

    def _route_single_order(self, order: dict, active_map: dict):
        """
        Route a single order to an account.

        :param order: order row from database
        """
        order_id = order["id"]
        direction = order["direction"]  # LONG or SHORT
        kelly_rate = float(order["kelly_rate"])
        leverage = int(order["leverage"] or 1)

        logger.info(f"Routing order {order_id}, direction={direction}, kelly_rate={kelly_rate}, leverage={leverage}")

        selected_account = None
        min_available_balance = float("inf")

        total_asset_btc = ClientRegistry.get_total_asset_btc_sum()
        qty = scale_quantity(kelly_rate * total_asset_btc)

        account_clients = ClientRegistry.all_clients()
        
        if qty <= 0:
            logger.warning(f"Calculated order quantity= {qty} is zero or negative for order {order_id}, mark FAILED")
            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_FAILED,
                failed_reason=f"Calculated order quantity= {qty} is zero or negative"
            )
            return

        for account_name, client in account_clients.items():
            try:
                # 0️⃣ skip if account already has active order in this direction
                if active_map.get((account_name, direction)):
                    logger.info(f"[{account_name}] already has active {direction} order, skip")
                    continue

                # 1️⃣ direction position must be zero
                pos = client.current_position  # {'LONG': x, 'SHORT': y}
                if pos.get(direction, 0.0) != 0.0:
                    logger.info(f"[{account_name}] has {direction} position, skip")
                    continue

                # 2️⃣ check balance
                mark_price = client.get_mark_price()
                available_balance = client.available_balance

                required_margin = (qty * mark_price / leverage) * MARGIN_SAFETY_FACTOR

                if available_balance < required_margin:
                    logger.info(
                    f"[{account_name}] insufficient USDT balance "
                    f"(avail={available_balance:.2f}, need={required_margin:.2f})"
                )
                    continue

            # 3️⃣ choose smallest available balance (most conservative)
                if available_balance < min_available_balance:
                    min_available_balance = available_balance
                    selected_account = account_name

            except Exception as e:
                logger.warning(f"Failed to evaluate account {account_name}: {e}")

        if not selected_account:
            logger.warning(f"No eligible account for order {order_id}, mark FAILED")
            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_FAILED,
                failed_reason=FAILED_REASON_NO_AVAILABLE_ACCOUNT
            )
            return

        logger.info(f"Order {order_id} assigned to account {selected_account}")

        self.repo.update_order(
            order_id,
            status=ORDER_STATUS_ASSIGNED,
            assigned_account=selected_account,
            quantity=qty
        )

        # update active_map so next orders see this account as busy
        active_map.setdefault((selected_account, direction), []).append(order)