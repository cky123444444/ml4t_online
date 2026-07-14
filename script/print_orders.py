import sqlite3
import argparse
from datetime import datetime, timedelta, timezone
from tabulate import tabulate

ORDER_DB_PATH = "/home/chenkeyi_sg/ML_server_demo/data/orders.db"


def parse_time(ts_str):
    if ts_str is None:
        return None
    return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)


def format_time(ts):
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def format_duration(start_ts, end_ts):
    if not start_ts or not end_ts:
        return "-"
    duration_sec = end_ts - start_ts
    duration_min = duration_sec // 60
    return f"{duration_min} min"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=ORDER_DB_PATH, help="SQLite DB path")
    parser.add_argument("--start", help="start time ISO format (e.g. 2026-03-09T00:00)")
    parser.add_argument("--end", help="end time ISO format")

    args = parser.parse_args()

    now = datetime.now(timezone.utc)

    start = parse_time(args.start)
    end = parse_time(args.end)

    if start is None:
        start = now - timedelta(days=1)

    if end is None:
        end = now

    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT
            id,
            assigned_account,
            symbol,
            direction,
            quantity,
            entry_price,
            exit_price,
            status,
            commission,
            created_at,
            close_time
        FROM orders
        WHERE created_at >= ? AND created_at <= ?
        ORDER BY created_at
        """,
        (start_ts, end_ts),
    ).fetchall()

    print("\nDB:", args.db)
    print("Time Range:", start, "→", end)
    print("Total rows:", len(rows))
    print()

    if not rows:
        return

    table = []

    for r in rows:
        table.append(
            [
                r["id"],
                r["assigned_account"] or "-",
                r["symbol"],
                r["direction"],
                r["quantity"],
                r["entry_price"] or "-",
                r["exit_price"] or "-",
                r["status"],
                r["commission"] or "-",
                format_time(r["created_at"]),
                format_time(r["close_time"]),
                (r["exit_price"] - r["entry_price"])*r["quantity"] - r["commission"] if r["direction"] == "LONG" else (r["entry_price"] - r["exit_price"])*r["quantity"] - r["commission"],
                format_duration(r["created_at"], r["close_time"])
            ]
        )

    headers = [
        "id",
        "account",
        "symbol",
        "dir",
        "qty",
        "entry",
        "exit",
        "status",
        "fee",
        "created_at",
        "closed_at",
        "pnl",
        "hold_duration"
    ]

    print(tabulate(table, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()