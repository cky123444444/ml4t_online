from src.storage.order_repo import OrderRepo
from src.ops.clients.client_registry import ClientRegistry
from src.utils.logger import setup_logger
from src.utils.calculate_helper import calc_avg_price, calc_commission
from src.config.constants import ORDER_STATUS_FAILED, \
    FAILED_REASON_MAX_RETRY, ORDER_STATUS_CLOSED, \
    FAILED_REASON_ASSIGNED_ACCOUNT_NOT_FOUND, ORDER_STATUS_SETTLED

logger = setup_logger("order_finalizer", log_file_name="executor.log")

class OrderFinalizer:
    def __init__(self, max_retry: int = 5):
        """
        OrderFinalizer enriches CLOSED orders with real execution data
        and marks them as SETTLED.
        """
        self.repo = OrderRepo()
        self.max_retry = max_retry

    def finalize_orders(self):
        orders = self.repo.fetch_closed_orders()

        if not orders:
            # logger.info("No CLOSED orders to settle")
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
                self._finalize_single_order(order)
            except Exception as e:
                logger.exception(
                    f"Unexpected error while finalizing order {order_id}: {e}"
                )   


    def _finalize_single_order(self, order: dict):
        order_id = order["id"]
        account_name = order["assigned_account"]
        retry_count = order.get("retry_count", 0)

        client = ClientRegistry.get_client(account_name)

        if not client:
            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_CLOSED,
                retry_count=retry_count+1,
                failed_reason=FAILED_REASON_ASSIGNED_ACCOUNT_NOT_FOUND
            )
            return

        try:
            entry_trades = client.get_order_trades(order.get("entry_order_id"))
            exit_trades = client.get_order_trades(order.get("exit_order_id"))

            # Check if trades are valid (empty list is treated as invalid)
            if not entry_trades or not exit_trades:
                raise Exception(f"Trades not ready or unavailable: entry_trades={entry_trades}, exit_trades={exit_trades}")

            entry_price = calc_avg_price(entry_trades)
            exit_price = calc_avg_price(exit_trades)
            commission = calc_commission(entry_trades) + calc_commission(exit_trades)

            # Check if prices are valid
            if entry_price <= 0 or exit_price <= 0:
                raise Exception(f"Invalid price detected: entry_price={entry_price}, exit_price={exit_price}")

            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_SETTLED,
                entry_price=entry_price,
                exit_price=exit_price,
                commission=commission,
                retry_count=0,
            )

            logger.info(f"Order {order_id} finalized successfully")

        except Exception as e:
            logger.exception(f"Failed to finalize order {order_id}: {e}")
            self.repo.update_order(
                order_id,
                status=ORDER_STATUS_CLOSED,
                retry_count=retry_count+1,
                failed_reason=f"{e}"
            )