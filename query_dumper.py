#!/usr/bin/env python3
"""
Feature Dumper Query Tool - HDF5 和 SQLite 统一查询

同时支持查询 HDF5 和 SQLite 中存储的特征数据。

使用示例:
    python query_dumper.py --stats              # 显示两个dumper的统计
    python query_dumper.py --hdf --stats        # 仅显示HDF统计
    python query_dumper.py --sql --stats        # 仅显示SQL统计
    python query_dumper.py --hdf --latest 5     # HDF最新5条
    python query_dumper.py --hdf --date 20240115  # 指定日期的HDF数据
    python query_dumper.py --sql --latest 5 --full  # SQL最新5条（完整）
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np


class HDF5Query:
    """HDF5 特征查询工具"""
    
    def __init__(self, hdf_dir: str = None):
        self.hdf_dir = hdf_dir or os.environ.get(
            'HDF_DUMP_DIR',
            '/app/data/hdf_dumper' if os.path.exists('/app/src') else './data/hdf_dumper'
        )
        
        if not os.path.exists(self.hdf_dir):
            raise FileNotFoundError(f"HDF 目录不存在: {self.hdf_dir}")
    
    def list_files(self) -> List[str]:
        """列出所有HDF5文件"""
        files = [f for f in os.listdir(self.hdf_dir) if f.endswith('.h5')]
        return sorted(files, reverse=True)  # 最新的在前
    
    def stats(self) -> Dict:
        """获取HDF5统计信息"""
        files = self.list_files()
        total_records = 0
        total_size = 0
        symbols = set()
        models = set()
        date_range = {'first': None, 'last': None}
        
        for file in files:
            file_path = os.path.join(self.hdf_dir, file)
            try:
                with pd.HDFStore(file_path, mode='r') as store:
                    # 读取元数据
                    if 'features/metadata' in store:
                        metadata = store['features/metadata']
                        total_records += len(metadata)
                        
                        # 收集symbols和models
                        if 'symbol' in metadata.columns:
                            symbols.update(metadata['symbol'].unique())
                        if 'model_name' in metadata.columns:
                            models.update(metadata['model_name'].unique())
                        
                        # 时间范围
                        if 'timestamp' in metadata.columns:
                            timestamps = pd.to_datetime(metadata['timestamp'])
                            if date_range['first'] is None:
                                date_range['first'] = timestamps.min().isoformat()
                            date_range['last'] = timestamps.max().isoformat()
                
                total_size += os.path.getsize(file_path)
            except Exception as e:
                print(f"警告: 读取 {file} 失败: {e}", file=sys.stderr)
        
        return {
            'hdf_dir': self.hdf_dir,
            'total_files': len(files),
            'total_records': total_records,
            'total_size_bytes': total_size,
            'total_size_mb': f"{total_size / 1024 / 1024:.2f} MB",
            'symbols': sorted(list(symbols)),
            'models': sorted(list(models)),
            'date_range': date_range
        }
    
    def get_daily_stats(self, date_str: str) -> Dict:
        """获取指定日期的统计"""
        # date_str 格式: YYYYMMDD
        file_name = f"features_{date_str}.h5"
        file_path = os.path.join(self.hdf_dir, file_name)
        
        if not os.path.exists(file_path):
            return {'error': f'文件不存在: {file_name}', 'date': date_str}
        
        try:
            with pd.HDFStore(file_path, mode='r') as store:
                if 'features/metadata' not in store:
                    return {'error': '找不到元数据', 'date': date_str}
                
                metadata = store['features/metadata']
                
                return {
                    'date': date_str,
                    'file': file_name,
                    'file_size_bytes': os.path.getsize(file_path),
                    'file_size_mb': f"{os.path.getsize(file_path) / 1024 / 1024:.2f} MB",
                    'total_records': len(metadata),
                    'symbols': metadata['symbol'].unique().tolist() if 'symbol' in metadata else [],
                    'models': metadata['model_name'].unique().tolist() if 'model_name' in metadata else [],
                    'time_range': {
                        'start': metadata['timestamp'].min() if 'timestamp' in metadata else None,
                        'end': metadata['timestamp'].max() if 'timestamp' in metadata else None
                    }
                }
        except Exception as e:
            return {'error': str(e), 'date': date_str}
    
    def latest(self, n: int = 10, date_str: str = None) -> List[Dict]:
        """获取最新N条记录"""
        records = []
        
        if date_str:
            # 查询特定日期
            file_name = f"features_{date_str}.h5"
            file_path = os.path.join(self.hdf_dir, file_name)
            if not os.path.exists(file_path):
                return []
            files = [file_name]
        else:
            # 查询所有文件，按最新优先
            files = self.list_files()
        
        for file in files:
            file_path = os.path.join(self.hdf_dir, file)
            try:
                with pd.HDFStore(file_path, mode='r') as store:
                    if 'features/metadata' in store:
                        metadata = store['features/metadata']
                        # 按timestamp排序，最新的在前
                        if 'timestamp' in metadata.columns:
                            metadata = metadata.sort_values('timestamp', ascending=False)
                        
                        for _, row in metadata.iterrows():
                            records.append({
                                'request_id': row.get('request_id', ''),
                                'timestamp': str(row.get('timestamp', '')),
                                'symbol': row.get('symbol', ''),
                                'model_name': row.get('model_name', ''),
                                'file': file
                            })
                            if len(records) >= n:
                                return records[:n]
            except Exception as e:
                print(f"警告: 读取 {file} 失败: {e}", file=sys.stderr)
        
        return records[:n]
    
    def read_features(self, date_str: str, request_id: str = None) -> Dict:
        """读取特定日期的特征"""
        file_name = f"features_{date_str}.h5"
        file_path = os.path.join(self.hdf_dir, file_name)
        
        if not os.path.exists(file_path):
            return {'error': f'文件不存在: {file_name}'}
        
        try:
            with pd.HDFStore(file_path, mode='r') as store:
                metadata = store['features/metadata'] if 'features/metadata' in store else None
                features = store['features/matrix'] if 'features/matrix' in store else None
                
                if request_id and metadata is not None:
                    # 按request_id过滤
                    mask = metadata['request_id'] == request_id
                    if not mask.any():
                        return {'error': f'未找到request_id: {request_id}'}
                    
                    idx = metadata[mask].index[0]
                    return {
                        'metadata': metadata.loc[[idx]].to_dict(orient='records')[0],
                        'features': features.iloc[[idx]].values.tolist() if features is not None else None
                    }
                
                return {
                    'metadata_count': len(metadata) if metadata is not None else 0,
                    'features_shape': features.shape if features is not None else None,
                    'date': date_str
                }
        except Exception as e:
            return {'error': str(e)}


class SQLQuery:
    """SQLite 特征查询工具"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.environ.get(
            'FEATURE_DUMP_DB_PATH',
            '/app/data/feature_dumper/feature_dump.db' if os.path.exists('/app/src') else './data/feature_dumper/feature_dump.db'
        )
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
    
    def _conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def stats(self) -> Dict:
        """获取SQL统计信息"""
        with self._conn() as conn:
            cur = conn.cursor()
            
            # 总数
            total = cur.execute("SELECT COUNT(*) FROM feature_records").fetchone()[0]
            
            # 按模型
            by_model = dict(cur.execute(
                "SELECT model_name, COUNT(*) FROM feature_records GROUP BY model_name"
            ).fetchall())
            
            # 按symbol
            by_symbol = dict(cur.execute(
                "SELECT symbol, COUNT(*) FROM feature_records GROUP BY symbol"
            ).fetchall())
            
            # 时间范围
            time_range = cur.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM feature_records"
            ).fetchone()
            
            # 文件大小
            size = os.path.getsize(self.db_path)
            
            return {
                'db_path': self.db_path,
                'total_records': total,
                'by_model': by_model,
                'by_symbol': by_symbol,
                'time_range': {
                    'start': time_range[0],
                    'end': time_range[1]
                },
                'db_size_bytes': size,
                'db_size_mb': f"{size / 1024 / 1024:.2f} MB"
            }
    
    def latest(self, n: int = 10) -> List[Dict]:
        """最新N条记录"""
        with self._conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT * FROM feature_records ORDER BY inserted_at DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(row) for row in rows]
    
    def by_request_id(self, request_id: str) -> Optional[Dict]:
        """按request_id查询"""
        with self._conn() as conn:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT * FROM feature_records WHERE request_id = ?", (request_id,)
            ).fetchone()
            return dict(row) if row else None


class DumperQuery:
    """统一查询接口"""
    
    def __init__(self, hdf_dir: str = None, db_path: str = None):
        self.hdf = None
        self.sql = None
        
        try:
            self.hdf = HDF5Query(hdf_dir)
        except FileNotFoundError as e:
            print(f"⚠️  HDF5: {e}", file=sys.stderr)
        
        try:
            self.sql = SQLQuery(db_path)
        except FileNotFoundError as e:
            print(f"⚠️  SQL: {e}", file=sys.stderr)
        
        if not self.hdf and not self.sql:
            raise FileNotFoundError("既找不到HDF5目录也找不到SQL数据库")


def print_hdf_stats(stats: Dict):
    """打印HDF5统计"""
    print("\n" + "="*80)
    print("📊 HDF5 统计信息")
    print("="*80)
    if 'error' in stats:
        print(f"❌ {stats['error']}")
        return
    
    print(f"📁 目录: {stats['hdf_dir']}")
    print(f"📄 文件数: {stats['total_files']}")
    print(f"📈 总记录: {stats['total_records']:,}")
    print(f"💾 总大小: {stats['total_size_mb']}")
    print(f"📍 交易对: {', '.join(stats['symbols']) if stats['symbols'] else '无'}")
    print(f"🤖 模型: {', '.join(stats['models']) if stats['models'] else '无'}")
    if stats['date_range']['first']:
        print(f"📅 时间范围: {stats['date_range']['first']} ~ {stats['date_range']['last']}")


def print_sql_stats(stats: Dict):
    """打印SQL统计"""
    print("\n" + "="*80)
    print("🗄️  SQLite 统计信息")
    print("="*80)
    if 'error' in stats:
        print(f"❌ {stats['error']}")
        return
    
    print(f"📊 数据库: {stats['db_path']}")
    print(f"📈 总记录: {stats['total_records']:,}")
    print(f"💾 数据库大小: {stats['db_size_mb']}")
    print(f"🤖 模型分布:")
    for model, count in stats['by_model'].items():
        print(f"     - {model}: {count:,}")
    print(f"📍 交易对分布:")
    for symbol, count in stats['by_symbol'].items():
        print(f"     - {symbol}: {count:,}")
    if stats['time_range']['start']:
        print(f"📅 时间范围: {stats['time_range']['start']} ~ {stats['time_range']['end']}")


def print_record(rec: Dict, full: bool = False, source: str = 'SQL'):
    """打印单条记录"""
    print("─" * 80)
    if source == 'HDF5':
        print(f"[HDF5] {rec.get('file', 'unknown')}")
        print(f"Request ID: {rec['request_id']}")
        print(f"Time: {rec['timestamp']} | Model: {rec['model_name']} | Symbol: {rec['symbol']}")
    else:  # SQL
        print(f"Request ID: {rec['request_id']}")
        print(f"Time: {rec['timestamp']} | Model: {rec['model_name']} | Symbol: {rec['symbol']}")
        
        if full:
            # OHLCV
            ohlcv_data = rec.get('ohlcv_data', '{}')
            try:
                ohlcv = json.loads(ohlcv_data) if isinstance(ohlcv_data, str) else ohlcv_data
                if isinstance(ohlcv, list):
                    print(f"\nOHLCV: {len(ohlcv)} 条数据")
                    if ohlcv:
                        print(f"  首: {ohlcv[0]}")
                        print(f"  尾: {ohlcv[-1]}")
                elif isinstance(ohlcv, dict):
                    print(f"\nOHLCV: {ohlcv}")
            except:
                pass
            
            # Features
            features_data = rec.get('features', '[]')
            try:
                features = json.loads(features_data) if isinstance(features_data, str) else features_data
                if isinstance(features, list) and features:
                    print(f"\nFeatures: {len(features)} × {len(features[0]) if features[0] else 0}")
                    if features[0]:
                        print(f"  首行前5个: {features[0][:5]}")
            except:
                pass
            
            # Model output
            model_output = rec.get('model_output', '{}')
            try:
                output = json.loads(model_output) if isinstance(model_output, str) else model_output
                print(f"\nModel Output: {json.dumps(output, ensure_ascii=False)}")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='Feature 查询工具 (HDF5 + SQLite 统一接口)')
    
    # 数据源选择
    source_group = parser.add_argument_group('数据源')
    source_group.add_argument('--hdf', action='store_true', help='仅查询HDF5')
    source_group.add_argument('--sql', action='store_true', help='仅查询SQLite')
    source_group.add_argument('--hdf-dir', help='HDF5目录路径')
    source_group.add_argument('--db', help='SQLite数据库路径')
    
    # 查询类型
    query_group = parser.add_argument_group('查询类型')
    query_group.add_argument('--stats', action='store_true', help='显示统计信息')
    query_group.add_argument('--latest', type=int, metavar='N', help='查询最新N条记录')
    query_group.add_argument('--date', help='HDF5 查询指定日期 (YYYYMMDD)')
    query_group.add_argument('--request-id', help='按request_id查询')
    
    # 显示选项
    display_group = parser.add_argument_group('显示选项')
    display_group.add_argument('--full', action='store_true', help='显示完整信息')
    
    args = parser.parse_args()
    
    try:
        # 默认查询两个源
        query_hdf = not args.sql
        query_sql = not args.hdf
        
        query = DumperQuery(hdf_dir=args.hdf_dir, db_path=args.db)
        
        # --stats 统计信息
        if args.stats:
            if query_hdf and query.hdf:
                print_hdf_stats(query.hdf.stats())
            if query_sql and query.sql:
                print_sql_stats(query.sql.stats())
            return
        
        # --date 查询HDF5特定日期
        if args.date:
            if not query.hdf:
                print("❌ HDF5不可用", file=sys.stderr)
                return
            
            stats = query.hdf.get_daily_stats(args.date)
            print("\n📅 HDF5 日期统计:")
            print(json.dumps(stats, ensure_ascii=False, indent=2))
            
            # 显示该日期的最新记录
            records = query.hdf.latest(n=5, date_str=args.date)
            if records:
                print(f"\n最新5条记录:")
                for rec in records:
                    print_record(rec, full=args.full, source='HDF5')
            return
        
        # --request-id 查询特定request
        if args.request_id:
            print(f"\n🔍 查询 request_id: {args.request_id}\n")
            
            if query_sql and query.sql:
                rec = query.sql.by_request_id(args.request_id)
                if rec:
                    print("📊 SQL 记录:")
                    print_record(rec, full=True, source='SQL')
            
            if query_hdf and query.hdf:
                # HDF5中按request_id查询比较困难，可以列出最新的并手动查找
                latest = query.hdf.latest(n=100)
                matching = [r for r in latest if r['request_id'] == args.request_id]
                if matching:
                    print("\n📊 HDF5 记录:")
                    for rec in matching[:1]:
                        print_record(rec, full=args.full, source='HDF5')
            return
        
        # --latest N 最新记录
        if args.latest:
            records = []
            
            if query_sql and query.sql:
                sql_records = query.sql.latest(args.latest)
                for rec in sql_records:
                    records.append((rec, 'SQL'))
            
            if query_hdf and query.hdf:
                hdf_records = query.hdf.latest(args.latest)
                for rec in hdf_records:
                    records.append((rec, 'HDF5'))
            
            # 按时间排序
            records.sort(key=lambda x: x[0].get('timestamp', ''), reverse=True)
            records = records[:args.latest]
            
            if not records:
                print("❌ 未找到记录")
                return
            
            print(f"\n📝 共 {len(records)} 条记录\n")
            for rec, source in records:
                print_record(rec, full=args.full, source=source)
            return
        
        # 默认显示统计
        if query_hdf and query.hdf:
            print_hdf_stats(query.hdf.stats())
        if query_sql and query.sql:
            print_sql_stats(query.sql.stats())
    
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()