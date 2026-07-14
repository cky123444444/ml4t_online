#!/usr/bin/env python3
"""
Feature Dumper Query Tool - 简洁版

查询 feature_dumper 存储的数据。

使用示例:
    python query_features.py --stats
    python query_features.py --latest 5
    python query_features.py --latest 5 --full
"""

import os
import sys
import json
import sqlite3
import argparse
from typing import List, Dict, Any, Optional


class FeatureQuery:
    """简洁的特征查询工具"""
    
    def __init__(self, db_path: str = None):
        if db_path:
            # 使用明确指定的路径
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"数据库文件不存在: {db_path}")
            self.db_path = db_path
        else:
            # 使用确定的默认路径（基于运行环境）
            # 在容器内: /app/data/feature_dumper/feature_dump.db
            # 在宿主机: ./data/feature_dumper/feature_dump.db（相对于项目根目录）
            default_paths = {
                'container': '/app/data/feature_dumper/feature_dump.db',
                'host': './data/feature_dumper/feature_dump.db'
            }
            
            # 判断是否在容器内（检查特征文件）
            is_container = os.path.exists('/app/src')
            default_path = default_paths['container'] if is_container else default_paths['host']
            
            # 也支持环境变量覆盖
            self.db_path = os.environ.get('FEATURE_DUMP_DB_PATH', default_path)
            
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(
                    f"数据库文件不存在: {self.db_path}\n"
                    f"预期路径: 容器内={default_paths['container']}, "
                    f"宿主机={default_paths['host']}\n"
                    f"请确保:\n"
                    f"  1. 已运行过预测请求\n"
                    f"  2. ENABLE_FEATURE_DUMP=1\n"
                    f"  3. docker-compose.yml 的 volume 映射正确"
                )
    
    def _conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def stats(self) -> Dict:
        """统计信息"""
        with self._conn() as conn:
            cur = conn.cursor()
            
            # 总数
            total = cur.execute("SELECT COUNT(*) FROM feature_records").fetchone()[0]
            
            # 按模型
            by_model = dict(cur.execute(
                "SELECT model_name, COUNT(*) FROM feature_records GROUP BY model_name"
            ).fetchall())
            
            # 时间范围
            time_range = cur.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM feature_records"
            ).fetchone()
            
            # 文件大小
            size = os.path.getsize(self.db_path)
            
            return {
                'db_path': self.db_path,
                'total': total,
                'by_model': by_model,
                'first_time': time_range[0],
                'last_time': time_range[1],
                'size_mb': f"{size / 1024 / 1024:.2f} MB"
            }
    
    def latest(self, n: int = 10) -> List[Dict]:
        """最新 N 条记录"""
        with self._conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT * FROM feature_records ORDER BY inserted_at DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(row) for row in rows]
    
    def by_request_id(self, request_id: str) -> Optional[Dict]:
        """按 request_id 查询"""
        with self._conn() as conn:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT * FROM feature_records WHERE request_id = ?", (request_id,)
            ).fetchone()
            return dict(row) if row else None


def print_stats(stats: Dict):
    """打印统计信息"""
    print(f"\n📊 数据库: {stats['db_path']}")
    print(f"   大小: {stats['size_mb']}")
    print(f"   总记录: {stats['total']:,}")
    print(f"   时间: {stats['first_time']} ~ {stats['last_time']}")
    print(f"   模型分布:")
    for model, count in stats['by_model'].items():
        print(f"     - {model}: {count:,}")


def print_record(rec: Dict, full: bool = False):
    """打印记录"""
    print("─" * 80)
    print(f"Request ID: {rec['request_id']}")
    print(f"Time: {rec['timestamp']} | Model: {rec['model_name']} | Symbol: {rec['symbol']}")
    
    if full:
        # OHLCV
        ohlcv = json.loads(rec['ohlcv_data'])
        if isinstance(ohlcv, list):
            print(f"\nOHLCV: {len(ohlcv)} 条数据")
            if ohlcv:
                print(f"  首: {ohlcv[0]}")
                print(f"  尾: {ohlcv[-1]}")
        
        # Features
        features = json.loads(rec['features'])
        if isinstance(features, list) and features:
            print(f"\nFeatures: {len(features)} × {len(features[0])}")
            print(f"  首行前5个: {features[0][:5]}")
        
        # Model output
        output = json.loads(rec['model_output'])
        print(f"\nModel Output:")
        print(f"  {json.dumps(output, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description='Feature 查询工具 (简洁版)')
    parser.add_argument('--db', help='数据库路径')
    parser.add_argument('--stats', action='store_true', help='显示统计')
    parser.add_argument('--latest', type=int, metavar='N', help='最新N条')
    parser.add_argument('--request-id', help='按 request_id 查询')
    parser.add_argument('--full', action='store_true', help='显示完整内容')
    
    args = parser.parse_args()
    
    try:
        q = FeatureQuery(db_path=args.db)
        
        if args.stats:
            print_stats(q.stats())
            return
        
        # 查询记录
        records = []
        if args.request_id:
            rec = q.by_request_id(args.request_id)
            if rec:
                records = [rec]
        elif args.latest:
            records = q.latest(args.latest)
        else:
            records = q.latest(10)
        
        if not records:
            print("❌ 未找到记录")
            return
        
        print(f"\n📝 共 {len(records)} 条记录\n")
        for rec in records:
            print_record(rec, full=args.full)
    
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        print("\n💡 解决方法:", file=sys.stderr)
        print("  1. 确认已发送预测请求: curl -X POST http://localhost:8000/predict -d '{\"model\":\"dragonnet\"}'", file=sys.stderr)
        print("  2. 在容器内查询: docker exec ml-server python query_features.py --stats", file=sys.stderr)
        print("  3. 手动指定路径: python query_features.py --db <path> --stats", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
