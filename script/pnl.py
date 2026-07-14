import sqlite3

ORDER_DB_PATH = "/home/chenkeyi_sg/ML_server_demo/data/orders.db"

conn = sqlite3.connect(ORDER_DB_PATH)
conn.row_factory = sqlite3.Row

query = """
SELECT
assigned_account,
SUM(
CASE
WHEN direction='LONG'
    THEN (exit_price-entry_price)*quantity
WHEN direction='SHORT'
    THEN (entry_price-exit_price)*quantity
END - IFNULL(commission,0)
) AS pnl,
COUNT(*) as trades
FROM orders
WHERE status IN ('SETTLED')
GROUP BY assigned_account
"""

rows = conn.execute(query).fetchall()

print("\nPNL by account\n")

total = 0
for r in rows:
    pnl = r["pnl"] or 0
    total += pnl
    print(f'{r["assigned_account"]:15} {pnl:12.4f}   trades={r["trades"]}')

print("\nTOTAL PNL:", round(total,4))