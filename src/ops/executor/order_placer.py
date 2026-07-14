from datetime import datetime, timezone

from src.storage.order_repo import OrderRepo
from src.ops.clients.client_registry import ClientRegistry
from src.utils.logger import setup_logger
from src.config.constants import ORDER_STATUS_ASSIGNED, ORDER_STATUS_FAILED, \
    FAILED_REASON_MAX_RETRY, FAILED_REASON_ASSIGNED_ACCOUNT_NOT_FOUND, \
    ORDER_STATUS_OPEN, SIDE_ACTION_OPEN_MAP

logger = setup_logger("order_placer", log_file_name="executor.log")

class OrderPlacer:
    """
    OrderPlacer sends ASSIGNED orders to exchange and manages retry logic.
    """

    def __init__(self, max_retry: int = 5):
        self.repo = OrderRepo()
        self.max_retry = max_retry

    def place_orders(self):
        orders = self.repo.fetch_assigned_orders()

        if not orders:
            # logger.info("No ASSIGNED orders to execute")
            return

        for order in orders:
            order_id = order["id"]
            retry_count = order.get("retry_count", 0)

            # ✅ retry policy decision happens HERE
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
                self._place_single_order(order)
            except Exception as e:
                logger.exception(
                    f"Unexpected error while executing order {order_id}: {e}"
                )

    def _place_single_order(self, order: dict):
        """
        Attempt to execute the order once.
        """
        order_id = order["id"]
        account_name = order["assigned_account"]
        direction = order["direction"]  # LONG / SHORT
        retry_count = order.get("retry_count", 0)

        logger.info(
            f"Executing order {order_id} on account {account_name}, retry={retry_count}"
        )

        client = ClientRegistry.get_client(account_name)
        if not client:
            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_FAILED,
                failed_reason=FAILED_REASON_ASSIGNED_ACCOUNT_NOT_FOUND
            )
            return

        try:
            result = client.place_order(
                side=SIDE_ACTION_OPEN_MAP[direction],
                quantity=order["quantity"],
                leverage=order["leverage"],
                position_side=order["direction"]
            )

            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_OPEN,
                entry_order_id=result["orderId"],
                entry_price=result.get("avgPrice", 0.0),
                open_time=int(datetime.now(timezone.utc).timestamp()),
                retry_count=0,
                failed_reason=None
            )

            logger.info(f"Order {order_id} submitted successfully")

        except Exception as e:
            logger.warning(
                f"Execution failed for order {order_id}, retrying: {e}"
            )

            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_ASSIGNED,
                retry_count=retry_count + 1,
                failed_reason=f"{e}"
            )
