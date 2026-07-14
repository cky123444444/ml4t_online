"""
HDF Dumper - Specialized dumper for HDF5 file format.

HDF5 is commonly used in ML training pipelines for storing large datasets.
This dumper writes feature data to HDF5 files for offline training.

Design Considerations:
- Pros: 
  * Efficient storage for numerical data (numpy arrays)
  * Fast read/write for training pipelines
  * Built-in compression support
  * Compatible with pandas, numpy, PyTorch, TensorFlow
- Cons:
  * Not suitable for concurrent writes from multiple processes
  * Requires careful file management (rotation, cleanup)
  * No ACID guarantees like databases
  * File corruption risk if process crashes during write

Use Cases:
- Batch collection for offline model training
- Feature engineering pipeline outputs
- Model evaluation datasets

File Organization:
- One HDF5 file per day (features_YYYYMMDD.h5)
- Automatic date-based rotation at midnight UTC
- Typical daily file size: ~1440 records (1 record/minute)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional
import json

from src.ops.dumper.base_dumper import BaseDumper
from src.utils.logger import setup_logger

logger = setup_logger('hdf_dumper')


class HDFDumper(BaseDumper[Dict[str, Any]]):
    """HDF5 dumper with daily file rotation."""
    
    def __init__(
        self,
        hdf_dir: Optional[str] = None,
        key: str = 'features',
        queue_size: int = 10000,
        batch_size: int = 1000,
        flush_interval: float = 10.0,
        compression: str = 'blosc',
        complevel: int = 5
    ):
        """
        Initialize HDF dumper with daily file rotation.
        
        Args:
            hdf_dir: Directory for HDF5 files
            key: HDF5 key prefix for data storage
            queue_size: Queue size for async writes
            batch_size: Batch size for writing
            flush_interval: Flush interval in seconds
            compression: Compression algorithm
                ✅ Supported: 'blosc', 'zlib', 'lzo', 'bzip2', 'blosc2'
                ❌ Not supported: 'gzip' (use 'zlib' instead)
                Default: 'blosc' (fastest)
            complevel: Compression level (0-9)
        """
        # ✅ 先设置属性（在调用 super().__init__() 之前）
        self.hdf_dir = hdf_dir or os.environ.get('HDF_DUMP_DIR', '/app/data/hdf_dumper')
        self.key = key
        self.compression = compression
        self.complevel = complevel
        
        # 当前文件状态
        self.current_file_path: Optional[str] = None
        self.current_file_records = 0
        self.current_date: Optional[str] = None
        
        # ✅ 调用父类构造器（会调用 _init_backend()）
        super().__init__(
            name='HDFDumper',
            queue_size=queue_size,
            batch_size=batch_size,
            flush_interval=flush_interval
        )
    
    def _init_backend(self):
        """
        ✅ 实现抽象方法：初始化 HDF5 后端
        
        这个方法会在 BaseDumper.__init__() 中被调用
        """
        try:
            # 创建 HDF5 目录
            os.makedirs(self.hdf_dir, exist_ok=True)
            
            # 初始化当前文件
            self._ensure_current_file()
            
            logger.info(f"HDF5 backend initialized: {self.hdf_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize HDF5 backend: {e}", exc_info=True)
            raise

    def _ensure_current_file(self):
        """确保当前文件是今天的日期文件，如果日期变更则自动切换"""
        today = datetime.now(timezone.utc).strftime('%Y%m%d')
        
        # 如果日期变了或首次初始化，切换到新文件
        if self.current_date != today:
            self.current_date = today
            self.current_file_path = os.path.join(
                self.hdf_dir,
                f"features_{today}.h5"
            )
            
            # 检查文件是否存在，读取已有记录数
            if os.path.exists(self.current_file_path):
                try:
                    with pd.HDFStore(self.current_file_path, mode='r') as store:
                        metadata_key = f'{self.key}/metadata'
                        if metadata_key in store:
                            self.current_file_records = len(store[metadata_key])
                            logger.info(f"Resumed daily file: {os.path.basename(self.current_file_path)}, "
                                      f"existing records: {self.current_file_records}")
                        else:
                            self.current_file_records = 0
                except Exception as e:
                    logger.warning(f"Failed to read existing file: {e}")
                    self.current_file_records = 0
            else:
                self.current_file_records = 0
                logger.info(f"Created new daily file: {os.path.basename(self.current_file_path)}")

    def _write_batch(self, records: List[Dict[str, Any]]) -> bool:
        """Write dict records to HDF5 with hierarchical structure and native arrays."""
        if not records:
            return True
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 每次写入前检查日期，自动切换文件
                self._ensure_current_file()
                
                # Separate metadata and numerical data
                metadata_records = []
                feature_matrices = []
                
                for record in records:
                    # Metadata (small structured data)
                    metadata = {
                        'request_id': record.get('request_id', ''),
                        'timestamp': record.get('timestamp', ''),
                        'symbol': record.get('symbol', ''),
                        'model_name': record.get('model_name', ''),
                        'created_at': record.get('created_at', ''),
                        'ohlcv_data': self._safe_json_dumps(record.get('ohlcv_data')),
                        'model_output': self._safe_json_dumps(record.get('model_output'))
                    }
                    metadata_records.append(metadata)
                    
                    # Features (large numerical arrays)
                    features = record.get('features')
                    if isinstance(features, list):
                        features = np.array(features, dtype=np.float32)
                    elif isinstance(features, np.ndarray):
                        features = features.astype(np.float32)
                    else:
                        features = np.array([], dtype=np.float32)
                    
                    if features.ndim == 1:
                        features = features.reshape(1, -1)
                    elif features.ndim > 2:
                        features = features.reshape(features.shape[0], -1)
                    
                    feature_matrices.append(features)
                
                # 写入数据
                self._write_hierarchical_data(metadata_records, feature_matrices)
                self.current_file_records += len(metadata_records)
                
                logger.debug(f"Wrote {len(metadata_records)} records to {os.path.basename(self.current_file_path)}, "
                           f"total: {self.current_file_records}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to write batch (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                if attempt == max_retries - 1:
                    return False
                # 重试前重新检查文件
                self._ensure_current_file()
        
        return False
    
    def get_daily_stats(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指定日期的统计信息
        
        Args:
            date_str: 日期字符串 (YYYYMMDD)，默认为今天
            
        Returns:
            统计信息字典
        """
        date_str = date_str or datetime.now(timezone.utc).strftime('%Y%m%d')
        file_path = os.path.join(self.hdf_dir, f"features_{date_str}.h5")
        
        if not os.path.exists(file_path):
            return {'error': 'File not found', 'date': date_str}
        
        try:
            with pd.HDFStore(file_path, mode='r') as store:
                metadata_key = f'{self.key}/metadata'
                if metadata_key in store:
                    metadata = store[metadata_key]
                    return {
                        'date': date_str,
                        'total_records': len(metadata),
                        'file_size': os.path.getsize(file_path),
                        'file_size_human': self._format_size(os.path.getsize(file_path)),
                        'symbols': metadata['symbol'].unique().tolist() if 'symbol' in metadata else [],
                        'models': metadata['model_name'].unique().tolist() if 'model_name' in metadata else [],
                        'time_range': {
                            'start': metadata['timestamp'].min() if 'timestamp' in metadata else None,
                            'end': metadata['timestamp'].max() if 'timestamp' in metadata else None
                        }
                    }
        except Exception as e:
            return {'error': str(e), 'date': date_str}
    
    def _write_hierarchical_data(
        self, 
        metadata_records: List[Dict[str, Any]], 
        feature_matrices: List[np.ndarray]
    ):
        """Write data using hierarchical structure for optimal storage and access.
        
        Structure:
        /features/metadata - Metadata table (queryable)
        /features/matrix   - Feature arrays (efficient numerical storage)
        """
        if not metadata_records:
            return
        
        metadata_df = pd.DataFrame(metadata_records)
        
        # Concatenate all feature matrices into single array
        features_array = np.vstack(feature_matrices)
        
        with pd.HDFStore(
            self.current_file_path,
            mode='a',
            complevel=self.complevel if self.compression else 0,
            complib=self.compression if self.compression else None
        ) as store:
            # Write metadata table (small, queryable)
            metadata_key = f'{self.key}/metadata'
            if metadata_key in store:
                store.append(
                    metadata_key,
                    metadata_df,
                    format='table',
                    data_columns=['symbol', 'model_name', 'timestamp'],
                    index=False,
                    min_itemsize={
                        'request_id': 50,
                        'symbol': 20,
                        'model_name': 30,
                        'timestamp': 30,
                        'created_at': 30,
                        'ohlcv_data': 30000,
                        'model_output': 5000
                    }
                )
            else:
                store.put(
                    metadata_key,
                    metadata_df,
                    format='table',
                    data_columns=['symbol', 'model_name', 'timestamp'],
                    index=False,
                    min_itemsize={
                        'request_id': 50,
                        'symbol': 20,
                        'model_name': 30,
                        'timestamp': 30,
                        'created_at': 30,
                        'ohlcv_data': 30000,
                        'model_output': 5000
                    }
                )
            
            # Write feature matrix
            matrix_key = f'{self.key}/matrix'
            features_df = pd.DataFrame(features_array)
            
            if matrix_key in store:
                store.append(matrix_key, features_df, format='table', index=False)
            else:
                store.put(matrix_key, features_df, format='table')
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate dict record."""
        return 'request_id' in record and 'symbol' in record and record['request_id'] and record['symbol']
    
    def dump_features(
        self,
        request_id: str,
        timestamp: datetime,
        symbol: str,
        ohlcv_data: Any,
        features: Any,
        model_name: str,
        model_output: Any,
        block: bool = False
    ) -> bool:
        """
        Dump features with proper handling of numerical arrays.
        
        Args:
            request_id: Unique request identifier
            timestamp: Prediction timestamp
            symbol: Trading symbol
            ohlcv_data: Raw OHLCV data
            features: Feature matrix (list or numpy array)
            model_name: Model name
            model_output: Model output
            block: Block if queue is full
            
        Returns:
            True if queued successfully
        """
        if isinstance(features, list):
            try:
                features = np.array(features)
            except Exception as e:
                logger.warning(f"Failed to convert features to numpy array: {e}")
        
        record = {
            'request_id': request_id,
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            'symbol': symbol,
            'model_name': model_name,
            'ohlcv_data': ohlcv_data,
            'features': features,
            'model_output': model_output,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        return self.dump(record, block=block)
    
    def _safe_json_dumps(self, obj: Any) -> str:
        """Safely serialize to JSON with proper handling of numpy arrays."""
        try:
            if obj is None:
                return "null"
            
            if isinstance(obj, np.ndarray):
                obj = obj.tolist()
            
            if isinstance(obj, list):
                obj = [x.tolist() if isinstance(x, np.ndarray) else x for x in obj]
                obj = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in obj]
            
            if hasattr(obj, 'to_dict'):
                obj = obj.to_dict()
            
            return json.dumps(obj, default=self._json_default, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON serialization failed: {e}")
            return json.dumps({"error": str(e), "type": str(type(obj))})
    
    def _json_default(self, obj):
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    
    def get_file_stats(self) -> Dict[str, Any]:
        """Get HDF5 file statistics."""
        stats = self.get_metrics()
        
        try:
            hdf_files = [f for f in os.listdir(self.hdf_dir) if f.endswith('.h5')]
            stats['total_files'] = len(hdf_files)
            
            total_size = sum(
                os.path.getsize(os.path.join(self.hdf_dir, f))
                for f in hdf_files
            )
            stats['total_size_bytes'] = total_size
            stats['total_size_human'] = self._format_size(total_size)
            
            if self.current_file_path and os.path.exists(self.current_file_path):
                stats['current_file'] = os.path.basename(self.current_file_path)
                stats['current_file_date'] = self.current_date
                stats['current_file_records'] = self.current_file_records
                stats['current_file_size'] = self._format_size(os.path.getsize(self.current_file_path))
            
        except Exception as e:
            logger.error(f"Failed to get file stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable string."""
        size: float = size_bytes
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"
    
    def list_files(self) -> List[str]:
        """List all HDF5 files sorted by date."""
        try:
            return sorted([
                os.path.join(self.hdf_dir, f)
                for f in os.listdir(self.hdf_dir)
                if f.endswith('.h5')
            ], reverse=True)  # 最新的在前
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def read_metadata(self, file_path: Optional[str] = None, query: Optional[str] = None) -> pd.DataFrame:
        """Read metadata from HDF5 file with optional query.
        
        Args:
            file_path: Path to HDF5 file (default: current file)
            query: Query string (e.g., "symbol == 'BTCUSDT' & model_name == 'dragonnet'")
            
        Returns:
            DataFrame with metadata
        """
        file_path = file_path or self.current_file_path
        if not file_path or not os.path.exists(file_path):
            return pd.DataFrame()
        
        try:
            with pd.HDFStore(file_path, mode='r') as store:
                metadata_key = f'{self.key}/metadata'
                if metadata_key in store:
                    if query:
                        return store.select(metadata_key, where=query)
                    else:
                        return store[metadata_key]
        except Exception as e:
            logger.error(f"Failed to read metadata: {e}")
        
        return pd.DataFrame()
    
    def read_features(self, file_path: Optional[str] = None, indices: Optional[List[int]] = None) -> np.ndarray:
        """Read feature matrix from HDF5 file.
        
        Args:
            file_path: Path to HDF5 file (default: current file)
            indices: Row indices to read (default: all)
            
        Returns:
            Feature matrix as numpy array
        """
        file_path = file_path or self.current_file_path
        if not file_path or not os.path.exists(file_path):
            return np.array([])
        
        try:
            with pd.HDFStore(file_path, mode='r') as store:
                matrix_key = f'{self.key}/matrix'
                if matrix_key in store:
                    matrix_df = store[matrix_key]
                    matrix = matrix_df.values
                    if indices is not None:
                        return matrix[indices]
                    return matrix
        except Exception as e:
            logger.error(f"Failed to read features: {e}")
        
        return np.array([])


# Global instance
_hdf_dumper: Optional[HDFDumper] = None


def get_hdf_dumper(**kwargs) -> HDFDumper:
    """Get or create global HDF dumper instance."""
    global _hdf_dumper
    if _hdf_dumper is None:
        _hdf_dumper = HDFDumper(**kwargs)
    return _hdf_dumper


def dump_to_hdf(
    request_id: str,
    timestamp: datetime,
    symbol: str,
    ohlcv_data: Any,
    features: Any,
    model_name: str,
    model_output: Any
) -> bool:
    """Convenience function to dump features to HDF5."""
    return get_hdf_dumper().dump_features(
        request_id=request_id,
        timestamp=timestamp,
        symbol=symbol,
        ohlcv_data=ohlcv_data,
        features=features,
        model_name=model_name,
        model_output=model_output,
        block=False
    )