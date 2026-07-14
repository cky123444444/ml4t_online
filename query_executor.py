#!/usr/bin/env python3
"""
Executor DB Query Tool - 订单执行数据库统一查询

支持查询订单状态、执行历史、账户分配等信息。

使用示例:
    python query_executor.py --stats                    # 显示统计信息
    python query_executor.py --latest 10                # 最新10条订单
    python query_executor.py --status OPEN              # 查询状态为OPEN的订单
    python query_executor.py --symbol BTCUSDT           # 查询特定交易对
    python query_executor.py --order-id 123 --full      # 完整显示订单详情
    python query_executor.py --date-range 20250101 20250115  # 时间范围查询
    python query_executor.py --account binance_1        # 按账户查询
"""

import os
import sys
import sqlite3
import argparse
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

from src.utils.logger import setup_logger, set_request_id

logger = setup_logger('query_executor')


class ExecutorQuery:
    """Executor DB 查询工具"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.environ.get(
            'EXECUTOR_DB_PATH',
            '/app/data/orders.db' if os.path.exists('/app/src') else './data/orders.db'
        )
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
        
        logger.info(f"使用数据库: {self.db_path}")
    
    def _conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def stats(self) -> Dict:
        """获取数据库统计信息"""
        with self._conn() as conn:
            cur = conn.cursor()
            
            # 总订单数
            total_orders = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            
            # 按状态分布
            by_status = dict(cur.execute(
                "SELECT status, COUNT(*) as count FROM orders GROUP BY status ORDER BY count DESC"
            ).fetchall())
            
            # 按方向分布
            by_direction = dict(cur.execute(
                "SELECT direction, COUNT(*) as count FROM orders WHERE direction IS NOT NULL GROUP BY direction"
            ).fetchall())
            
            # 按交易对分布
            by_symbol = dict(cur.execute(
                "SELECT symbol, COUNT(*) as count FROM orders GROUP BY symbol ORDER BY count DESC"
            ).fetchall())
            
            # 按账户分布
            by_account = dict(cur.execute(
                "SELECT assigned_account, COUNT(*) as count FROM orders WHERE assigned_account IS NOT NULL GROUP BY assigned_account"
            ).fetchall())
            
            # 活跃订单数 (ASSIGNED + OPEN)
            active_orders = cur.execute(
                "SELECT COUNT(*) FROM orders WHERE status IN ('ASSIGNED', 'OPEN')"
            ).fetchone()[0]
            
            # 已平仓订单数
            closed_orders = cur.execute(
                "SELECT COUNT(*) FROM orders WHERE status = 'CLOSED'"
            ).fetchone()[0]
            
            # 失败订单数
            failed_orders = cur.execute(
                "SELECT COUNT(*) FROM orders WHERE status = 'FAILED'"
            ).fetchone()[0]
            
            # 时间范围
            time_range = cur.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM orders"
            ).fetchone()
            
            # 盈亏统计
            pnl_stats = cur.execute("""
                SELECT 
                    COUNT(*) as count,
                    AVG(commission) as avg_commission,
                    SUM(CASE WHEN entry_price > 0 AND exit_price > 0 
                             THEN (exit_price - entry_price) * quantity 
                        END) as total_pnl
                FROM orders 
                WHERE entry_price > 0 AND exit_price > 0
            """).fetchone()
            
            # 数据库大小
            size = os.path.getsize(self.db_path)
            
            return {
                'db_path': self.db_path,
                'total_records': total_orders,
                'active_orders': active_orders,
                'closed_orders': closed_orders,
                'failed_orders': failed_orders,
                'by_status': by_status,
                'by_direction': by_direction,
                'by_symbol': by_symbol,
                'by_account': by_account,
                'time_range': {
                    'start': datetime.fromtimestamp(time_range[0]).isoformat() if time_range[0] else None,
                    'end': datetime.fromtimestamp(time_range[1]).isoformat() if time_range[1] else None
                },
                'pnl': {
                    'closed_count': pnl_stats[0] or 0,
                    'avg_commission': round(pnl_stats[1] or 0, 8),
                    'total_pnl': round(pnl_stats[2] or 0, 2)
                },
                'db_size_bytes': size,
                'db_size_mb': f"{size / 1024 / 1024:.2f} MB"
            }
    
    def latest(self, n: int = 10, status: str = None, symbol: str = None) -> List[Dict]:
        """查询最新N条订单"""
        with self._conn() as conn:
            cur = conn.cursor()
            
            query = "SELECT * FROM orders"
            params = []
            
            # 构建WHERE条件
            conditions = []
            if status:
                conditions.append("status = ?")
                params.append(status)
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(n)
            
            rows = cur.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    def by_status(self, status: str) -> List[Dict]:
        """按状态查询订单"""
        valid_statuses = ['NEW', 'ASSIGNED', 'OPEN', 'CLOSED', 'FAILED', 'SETTLED']
        if status not in valid_statuses:
            raise ValueError(f"无效状态: {status}，有效值: {valid_statuses}")
        
        with self._conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC",
                (status,)
            ).fetchall()
            return [dict(row) for row in rows]
    
    def by_symbol(self, symbol: str) -> List[Dict]:
        """按交易对查询订单"""
        with self._conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT * FROM orders WHERE symbol = ? ORDER BY created_at DESC",
                (symbol,)
            ).fetchall()
            return [dict(row) for row in rows]
    
    def by_account(self, account: str) -> List[Dict]:
        """按账户查询订单"""
        with self._conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT * FROM orders WHERE assigned_account = ? ORDER BY created_at DESC",
                (account,)
            ).fetchall()
            return [dict(row) for row in rows]
    
    def by_order_id(self, order_id: int) -> Optional[Dict]:
        """按订单ID查询"""
        with self._conn() as conn:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT * FROM orders WHERE id = ?",
                (order_id,)
            ).fetchone()
            return dict(row) if row else None
    
    def by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """时间范围查询 (YYYYMMDD 格式)"""
        try:
            start_ts = int(datetime.strptime(start_date, '%Y%m%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y%m%d').timestamp()) + 86400  # 加上一天
        except ValueError:
            raise ValueError("日期格式错误，请使用 YYYYMMDD")
        
        with self._conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT * FROM orders WHERE created_at BETWEEN ? AND ? ORDER BY created_at DESC",
                (start_ts, end_ts)
            ).fetchall()
            return [dict(row) for row in rows]
    
    def active_orders(self) -> List[Dict]:
        """获取所有活跃订单(ASSIGNED + OPEN)"""
        with self._conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT * FROM orders WHERE status IN ('ASSIGNED', 'OPEN') ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
    
    def failed_orders(self, limit: int = 100) -> List[Dict]:
        """获取失败订单"""
        with self._conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT * FROM orders WHERE status = 'FAILED' ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
    
    def get_account_stats(self, account: str) -> Dict:
        """获取账户统计"""
        with self._conn() as conn:
            cur = conn.cursor()
            
            stats = cur.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    SUM(CASE WHEN status IN ('ASSIGNED', 'OPEN') THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                    AVG(leverage) as avg_leverage,
                    SUM(CASE WHEN entry_price > 0 AND exit_price > 0 
                             THEN (exit_price - entry_price) * quantity 
                        END) as total_pnl
                FROM orders
                WHERE assigned_account = ?
            """, (account,)).fetchone()
            
            if not stats[0]:  # 没有订单
                return {'error': f'账户 {account} 无订单'}
            
            return {
                'account': account,
                'total_orders': stats[0],
                'active_orders': stats[1] or 0,
                'closed_orders': stats[2] or 0,
                'failed_orders': stats[3] or 0,
                'avg_leverage': round(stats[4] or 0, 1),
                'total_pnl': round(stats[5] or 0, 2)
            }


def format_timestamp(ts: int) -> str:
    """格式化时间戳"""
    if not ts:
        return 'N/A'
    try:
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(ts)


def print_stats(stats: Dict):
    """打印统计信息"""
    print("\n" + "="*80)
    print("📊 Executor DB 统计信息")
    print("="*80)
    if 'error' in stats:
        print(f"❌ {stats['error']}")
        return
    
    print(f"📊 数据库: {stats['db_path']}")
    print(f"💾 数据库大小: {stats['db_size_mb']}")
    print(f"\n📈 订单统计:")
    print(f"   总订单数: {stats['total_records']:,}")
    print(f"   活跃订单: {stats['active_orders']:,}")
    print(f"   已平仓: {stats['closed_orders']:,}")
    print(f"   失败: {stats['failed_orders']:,}")
    
    print(f"\n📍 订单状态分布:")
    for status, count in sorted(stats['by_status'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {status:10s}: {count:6,}")
    
    print(f"\n🔄 方向分布:")
    for direction, count in stats['by_direction'].items():
        print(f"   {direction}: {count:,}")
    
    print(f"\n💱 交易对 (Top 10):")
    for symbol, count in sorted(stats['by_symbol'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {symbol:12s}: {count:6,}")
    
    if stats['by_account']:
        print(f"\n💼 账户分布:")
        for account, count in sorted(stats['by_account'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {account}: {count:,}")
    
    if stats['time_range']['start']:
        print(f"\n📅 时间范围: {stats['time_range']['start']} ~ {stats['time_range']['end']}")
    
    print(f"\n💰 盈亏统计:")
    print(f"   已平仓订单: {stats['pnl']['closed_count']}")
    print(f"   平均费用: {stats['pnl']['avg_commission']}")
    print(f"   总P&L: {stats['pnl']['total_pnl']}")


def print_order(order: Dict, full: bool = False):
    """打印订单信息"""
    print("─" * 80)
    
    order_id = order.get('id', 'N/A')
    symbol = order.get('symbol', 'N/A')
    direction = order.get('direction', 'N/A')
    status = order.get('status', 'N/A')
    
    # 基本信息
    print(f"🆔 订单ID: {order_id} | 交易对: {symbol} | 方向: {direction} | 状态: {status}")
    print(f"⏰ 创建: {format_timestamp(order['created_at'])}")
    
    # 订单详情
    quantity = order.get('quantity')
    kelly_rate = order.get('kelly_rate')
    leverage = order.get('leverage')
    stop_loss = order.get('stop_loss')
    
    if quantity is not None:
        print(f"📊 数量: {quantity} | Kelly Rate: {kelly_rate} | 杠杆: {leverage}x | 止损: {stop_loss}")
    
    # 账户信息
    account = order.get('assigned_account')
    if account:
        print(f"💼 账户: {account}")
    
    # 订单ID
    entry_order_id = order.get('entry_order_id')
    exit_order_id = order.get('exit_order_id')
    if entry_order_id or exit_order_id:
        print(f"📝 开仓Order: {entry_order_id} | 平仓Order: {exit_order_id}")
    
    # 价格和时间
    entry_price = order.get('entry_price')
    exit_price = order.get('exit_price')
    open_time = order.get('open_time')
    close_time = order.get('close_time')
    
    if entry_price:
        print(f"💹 开仓价: {entry_price} @ {format_timestamp(open_time)}")
    if exit_price:
        print(f"💹 平仓价: {exit_price} @ {format_timestamp(close_time)}")
    
    # 费用和重试
    commission = order.get('commission')
    retry_count = order.get('retry_count')
    if commission is not None:
        print(f"💰 费用: {commission} | 重试: {retry_count}")
    
    # 失败原因
    failed_reason = order.get('failed_reason')
    if failed_reason:
        print(f"⚠️  失败原因: {failed_reason}")
    
    # 完整显示其他字段
    if full:
        print("\n📋 完整信息:")
        for key, value in order.items():
            if key not in ['id', 'symbol', 'direction', 'status', 'quantity', 'kelly_rate', 
                          'leverage', 'stop_loss', 'assigned_account', 'entry_order_id', 
                          'exit_order_id', 'entry_price', 'exit_price', 'open_time', 'close_time',
                          'commission', 'retry_count', 'failed_reason', 'created_at']:
                print(f"   {key}: {value}")


def print_orders(orders: List[Dict], full: bool = False):
    """打印多条订单"""
    if not orders:
        print("❌ 未找到订单")
        return
    
    print(f"\n📝 共 {len(orders)} 条订单\n")
    for order in orders:
        print_order(order, full=full)


def main():
    # 设置默认 request_id 用于标识 query_executor 工具
    set_request_id('query-exec')

    parser = argparse.ArgumentParser(description='Executor 数据库查询工具')
    
    # 数据库路径
    parser.add_argument('--db', help='数据库文件路径')
    
    # 查询类型
    query_group = parser.add_argument_group('查询类型')
    query_group.add_argument('--stats', action='store_true', help='显示统计信息')
    query_group.add_argument('--latest', type=int, metavar='N', help='最新N条订单')
    query_group.add_argument('--status', help='按状态查询 (NEW/ASSIGNED/OPEN/CLOSED/FAILED/SETTLED)')
    query_group.add_argument('--symbol', help='按交易对查询')
    query_group.add_argument('--order-id', type=int, help='按订单ID查询')
    query_group.add_argument('--account', help='按账户查询')
    query_group.add_argument('--date-range', nargs=2, metavar=('START', 'END'), help='时间范围查询 (YYYYMMDD YYYYMMDD)')
    query_group.add_argument('--active', action='store_true', help='显示所有活跃订单')
    query_group.add_argument('--failed', action='store_true', help='显示失败订单')
    
    # 显示选项
    display_group = parser.add_argument_group('显示选项')
    display_group.add_argument('--full', action='store_true', help='显示完整信息')
    
    args = parser.parse_args()
    
    try:
        query = ExecutorQuery(db_path=args.db)
        
        # --stats 统计信息
        if args.stats:
            print_stats(query.stats())
            return
        
        # --order-id 按ID查询
        if args.order_id:
            order = query.by_order_id(args.order_id)
            if order:
                print_order(order, full=True)
            else:
                print(f"❌ 订单不存在: {args.order_id}")
            return
        
        # --account 账户统计
        if args.account and not args.latest and not args.status and not args.symbol:
            stats = query.get_account_stats(args.account)
            print("\n" + "="*80)
            print(f"💼 账户统计: {stats['account']}")
            print("="*80)
            if 'error' not in stats:
                for key, value in stats.items():
                    if key != 'account':
                        print(f"   {key}: {value}")
            else:
                print(f"❌ {stats['error']}")
            return
        
        # --date-range 时间范围查询
        if args.date_range:
            orders = query.by_date_range(args.date_range[0], args.date_range[1])
            print_orders(orders, full=args.full)
            return
        
        # --active 活跃订单
        if args.active:
            orders = query.active_orders()
            print_orders(orders, full=args.full)
            return
        
        # --failed 失败订单
        if args.failed:
            orders = query.failed_orders()
            print_orders(orders, full=args.full)
            return
        
        # --latest N 最新N条
        if args.latest:
            orders = query.latest(n=args.latest, status=args.status, symbol=args.symbol)
            print_orders(orders, full=args.full)
            return
        
        # --status 按状态查询
        if args.status:
            orders = query.by_status(args.status)
            print_orders(orders, full=args.full)
            return
        
        # --symbol 按交易对查询
        if args.symbol:
            orders = query.by_symbol(args.symbol)
            print_orders(orders, full=args.full)
            return
        
        # 默认显示统计
        print_stats(query.stats())
    
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        print(f"❌ 参数错误: {e}", file=sys.stderr)
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        logger.error(str(e), exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
