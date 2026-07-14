"""
SQL Dumper - Specialized dumper for ML feature persistence.

Extends BaseDumper with ML-specific functionality:
- Feature data serialization
- SQLite backend with optimizations
- Feature statistics tracking
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional
from dataclasses import dataclass

from src.ops.dumper.base_dumper import BaseDumper
from src.utils.logger import setup_logger

logger = setup_logger('sql_dumper')


@dataclass
class FeatureRecord:
    """Data class for feature records."""
    request_id: str
    timestamp: str  # ISO format UTC timestamp
    symbol: str
    ohlcv_data: str  # JSON serialized
    features: str  # JSON serialized
    model_name: str
    model_output: str  # JSON serialized
    created_at: str


class SQLDumper(BaseDumper[FeatureRecord]):
    """
    SQL dumper with SQLite backend optimized for ML workloads.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 3.0
    ):
        """
        Initialize SQL dumper.
        
        Args:
            db_path: SQLite database path
            queue_size: Queue size
            batch_size: Batch size for writes
            flush_interval: Flush interval in seconds
        """
        self.db_path = db_path or os.environ.get(
            'FEATURE_DUMP_DB_PATH',
            '/app/data/feature_dumper/feature_dump.db'
        )
        
        # Create directory
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize base class
        super().__init__(
            name='SQLDumper',
            queue_size=queue_size,
            batch_size=batch_size,
            flush_interval=flush_interval
        )
    
    def _init_backend(self):
        """Initialize SQLite database with performance optimizations."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Performance optimizations
            cursor.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging
            cursor.execute('PRAGMA synchronous=NORMAL')  # Balance performance/safety
            cursor.execute('PRAGMA cache_size=10000')  # 10MB cache
            cursor.execute('PRAGMA temp_store=MEMORY')
            
            # Create table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    ohlcv_data TEXT,
                    features TEXT,
                    model_name TEXT,
                    model_output TEXT,
                    created_at TEXT NOT NULL,
                    inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON feature_records(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_request_id ON feature_records(request_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON feature_records(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON feature_records(model_name)')
            
            conn.commit()
            logger.info(f"SQLite database initialized at {self.db_path}")
            
        finally:
            conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path, timeout=30.0)
    
    def _write_batch(self, records: List[FeatureRecord]) -> bool:
        """Write a batch of feature records to SQLite."""
        if not records:
            return True
        
        conn = self._get_connection()
        try:
            conn.execute('BEGIN TRANSACTION')
            cursor = conn.cursor()
            
            cursor.executemany(
                '''
                INSERT INTO feature_records 
                (request_id, timestamp, symbol, ohlcv_data, features, model_name, model_output, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                [
                    (r.request_id, r.timestamp, r.symbol, r.ohlcv_data,
                     r.features, r.model_name, r.model_output, r.created_at)
                    for r in records
                ]
            )
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to write batch: {e}", exc_info=True)
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def _validate_record(self, record: FeatureRecord) -> bool:
        """Validate feature record."""
        if not record.request_id or not record.symbol:
            return False
        return True
    
    def dump_features(
        self,
        request_id: str,
        timestamp: Any,
        symbol: str,
        ohlcv_data: Any,
        features: List[List[float]],
        model_name: str,
        model_output: Any,
        block: bool = False
    ) -> bool:
        """
        Convenience method to dump feature data.
        
        Args:
            request_id: Unique request identifier
            timestamp: Prediction timestamp
            symbol: Trading symbol
            ohlcv_data: Raw OHLCV data
            features: Feature matrix
            model_name: Model name
            model_output: Model output
            block: Block if queue is full
            
        Returns:
            True if queued successfully
        """
        record = FeatureRecord(
            request_id=request_id,
            timestamp=timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            symbol=symbol,
            ohlcv_data=self._safe_json_dumps(ohlcv_data),
            features=self._safe_json_dumps(features),
            model_name=model_name,
            model_output=self._safe_json_dumps(model_output),
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        return self.dump(record, block=block)
    
    def _safe_json_dumps(self, obj: Any) -> str:
        """Safely serialize to JSON."""
        try:
            if obj is None:
                return "null"
            
            # Handle numpy/pandas
            if hasattr(obj, 'tolist'):
                obj = obj.tolist()
            elif hasattr(obj, 'to_dict'):
                obj = obj.to_dict()
            
            return json.dumps(obj, default=str, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"JSON serialization failed: {e}")
            return json.dumps({"error": str(e), "type": str(type(obj))})
    
    def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = self.get_metrics()
        
        try:
            if os.path.exists(self.db_path):
                stats['db_size_bytes'] = os.path.getsize(self.db_path)
                stats['db_size_human'] = self._format_size(stats['db_size_bytes'])
            
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM feature_records")
                stats['record_count'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM feature_records")
                stats['unique_symbols'] = cursor.fetchone()[0]
                
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _format_size(self, size_bytes: int) -> str:
        """Format byte size."""
        size: float = size_bytes
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"


# Global instance
_sql_dumper: Optional[SQLDumper] = None


def get_sql_dumper(**kwargs) -> SQLDumper:
    """Get or create global SQL dumper instance."""
    global _sql_dumper
    if _sql_dumper is None:
        _sql_dumper = SQLDumper(**kwargs)
    return _sql_dumper


def dump_to_sqlite(
    request_id: str,
    timestamp: datetime,
    symbol: str,
    ohlcv_data: Any,
    features: List[List[float]],
    model_name: str,
    model_output: Any
) -> bool:
    """Convenience function to dump features."""
    return get_sql_dumper().dump_features(
        request_id=request_id,
        timestamp=timestamp,
        symbol=symbol,
        ohlcv_data=ohlcv_data,
        features=features,
        model_name=model_name,
        model_output=model_output,
        block=False
    )