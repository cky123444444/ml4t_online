from datetime import datetime, timezone
from sqlite3 import Row
import os
from src.storage.db import get_conn
from src.utils.logger import setup_logger

logger = setup_logger("order_repo")


class OrderRepo:
    def __init__(self):
        # ✅ 使用环境变量，默认值为 /app/data/orders.db
        db_path = os.environ.get('ORDER_DB_PATH', '/app/data/orders.db')
        self.conn = get_conn(db_path=db_path)
        self.conn.row_factory = Row
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
            direction TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
            quantity REAL,
            kelly_rate REAL NOT NULL,
            leverage INTEGER,
            stop_loss REAL,
            status TEXT NOT NULL DEFAULT 'NEW'
                          CHECK(status IN ('NEW', 'ASSIGNED', 'OPEN', 'CLOSED', 'FAILED', 'SETTLED')),
            assigned_account TEXT,
            entry_order_id TEXT,
            exit_order_id TEXT,

            open_time INTEGER,
            close_time INTEGER,
            entry_price REAL,
            exit_price REAL,
            commission REAL,
                          
            retry_count INTEGER DEFAULT 0,
            failed_reason TEXT,
            created_at INTEGER
        )
        """)
        self.conn.commit()

    def create_order(self, symbol, direction, kelly_rate, leverage, stop_loss):
        self.conn.execute("""
        INSERT INTO orders
        (symbol, direction, kelly_rate, leverage, stop_loss, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'NEW', ?)
        """, (symbol, direction, kelly_rate, leverage, stop_loss, int(datetime.now(timezone.utc).timestamp())))
        self.conn.commit()

    def fetch_new_orders(self):
        orders = self.conn.execute(
            "SELECT * FROM orders WHERE status='NEW'"
        ).fetchall()
        return [dict(row) for row in orders]

    def fetch_open_orders(self):
        orders = self.conn.execute(
            "SELECT * FROM orders WHERE status='OPEN'"
        ).fetchall()
        return [dict(row) for row in orders]

    def fetch_assigned_orders(self):
        orders = self.conn.execute(
            "SELECT * FROM orders WHERE status='ASSIGNED'"
        ).fetchall()
        return [dict(row) for row in orders]
    
    def fetch_closed_orders(self):
        orders = self.conn.execute(
            "SELECT * FROM orders WHERE status='CLOSED'"
        ).fetchall()
        return [dict(row) for row in orders]
    
    def fetch_active_orders(self):
        orders = self.conn.execute(
            "SELECT * FROM orders WHERE status IN ('ASSIGNED', 'OPEN')"
        ).fetchall()
        return [dict(row) for row in orders]

    def update_order(self, order_id, **fields):
        keys = ", ".join(f"{k}=?" for k in fields)
        values = list(fields.values())
        values.append(order_id)
        cur = self.conn.execute(
            f"UPDATE orders SET {keys} WHERE id=?", values
        )
        self.conn.commit()
        rowcount = cur.rowcount
        logger.info(f"Updated order, order_id: {order_id}, affected rows: {rowcount}, fields: {fields}")


    
    def increment_retry(self, order_id):
        self.conn.execute("""
            UPDATE orders
            SET retry_count = retry_count + 1
            WHERE id = ?
            """, (order_id,))
        self.conn.commit()

    def mark_failed(self, order_id, reason=None):
        self.conn.execute("""
            UPDATE orders
            SET status = 'FAILED',
            failed_reason = ?
            WHERE id = ?
        """, (reason, order_id))
        self.conn.commit()

    def create_new_order(self,symbol, side, qty, stop_loss):
        self.conn.execute("""
            INSERT INTO orders (
                symbol,
                side,
                qty,
                stop_loss,
                status,
                retry_count,
                created_at
            )
            VALUES (?, ?, ?, ?, 'NEW', 0, ?)
            """, (
                symbol,
                side,
                qty,
                stop_loss,
                int(datetime.now(timezone.utc).timestamp())
            ))