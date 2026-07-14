from datetime import datetime, timezone
from src.storage.order_repo import OrderRepo
from src.ops.clients.client_registry import ClientRegistry
from src.utils.logger import setup_logger
from src.config.constants import (
    ORDER_STATUS_OPEN,
    ORDER_STATUS_CLOSED,
    ORDER_STATUS_FAILED,
    SIDE_ACTION_CLOSE_MAP,
    FAILED_REASON_ASSIGNED_ACCOUNT_NOT_FOUND,
    FAILED_REASON_MAX_RETRY
)

logger = setup_logger("order_closer", log_file_name="executor.log")


class OrderCloser:
    """
    OrderCloser is responsible for closing OPEN orders when exit conditions are met.
    """

    def __init__(self, max_retry: int = 20):
        """
        :param max_retry: max retry attempts before marking order as FAILED
        """
        self.repo = OrderRepo()
        self.max_retry = max_retry

    def process_open_orders(self):
        """
        Process all OPEN orders.
        """
        orders = self.repo.fetch_open_orders()
        if not orders:
            #logger.info("No OPEN orders to process")
            return

        for order in orders:
            order_id = order["id"]
            retry_count = order.get("retry_count", 0)

            if retry_count >= self.max_retry:
                logger.warning(
                    f"Order {order_id} exceeded max retry ({retry_count}), marking FAILED"
                )
                self.repo.update_order(
                    order_id,
                    status=ORDER_STATUS_FAILED,
                    failed_reason=f"{FAILED_REASON_MAX_RETRY}.{order.get("failed_reason", "")}"
                )
                continue

            try:
                if self.should_close(order):
                    self._close_single_order(order)
            except Exception as e:
                logger.exception(f"Unexpected error processing order {order['id']}: {e}")

    # =========================
    # Core Logic
    # =========================

    def should_close(self, order: dict) -> bool:
        """
        Determine whether an order should be closed.
        """
        now = int(datetime.now(timezone.utc).timestamp())
        open_time = order["open_time"]

        # 1. Max holding time: 1 hour
        if now - open_time >= 3600:
            logger.info(f"Order {order['id']} exceeded max holding time")
            return True

        # 2. Stop loss
        stop_loss = order.get("stop_loss")
        if stop_loss:
            account_name = order["assigned_account"]
            client = ClientRegistry.get_client(account_name)
            if not client:
                return False

            mark_price = client.get_mark_price()
            direction = order["direction"]

            if direction == "LONG" and mark_price <= stop_loss:
                logger.info(f"Order {order['id']} hit stop loss (LONG), mark_price={mark_price},\
                            stop_loss={stop_loss}")
                return True

            if direction == "SHORT" and mark_price >= stop_loss:
                logger.info(f"Order {order['id']} hit stop loss (SHORT), mark_price={mark_price},\
                            stop_loss={stop_loss}")
                return True

        return False


    def _close_single_order(self, order: dict):
        """
        Close a single OPEN order.
        """
        order_id = order["id"]
        account_name = order["assigned_account"]
        direction = order["direction"]
        quantity = order["quantity"]
        retry_count = order.get("retry_count", 0)

        logger.info(f"Closing order {order_id} on account {account_name}")

        client = ClientRegistry.get_client(account_name)
        if not client:
            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_OPEN,
                retry_count=retry_count+1,
                failed_reason=FAILED_REASON_ASSIGNED_ACCOUNT_NOT_FOUND
            )
            return

        try:
            # 1. Place closing order (SELL + same positionSide)
            result = client.place_order(
                side=SIDE_ACTION_CLOSE_MAP[direction],
                quantity=quantity,
                position_side=direction
            )

            # 3. Update DB
            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_CLOSED,
                exit_order_id=result["orderId"],
                close_time=int(datetime.now(timezone.utc).timestamp()),
                retry_count=0
            )

            logger.info(
                f"Order {order_id} closed successfully"
            )

        except Exception as e:
            logger.warning(f"Failed to close order {order_id}: {e}")
            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_OPEN,
                retry_count=retry_count+1,
                failed_reason=f"{e}"
            )

    