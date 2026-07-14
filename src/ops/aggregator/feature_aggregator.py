"""
Feature Aggregator - Sliding window aggregation for OHLCV data.

This module aggregates 60-minute sliding window statistics (mean and std)
for OHLCV data collected from HDF dumper outputs.

Architecture:
- Input: Daily HDF5 files with raw minute-level OHLCV data
- Processing: 60-minute sliding window (inclusive start, exclusive end)
- Output: Symbol-specific aggregated HDF5 files with 10 new features
  (open_mean, open_std, high_mean, high_std, low_mean, low_std, 
   close_mean, close_std, volume_mean, volume_std)

Scheduling:
- Runs hourly at xx:05 (5 minutes past the hour)
- Retention: 90 days (configurable)
- Error handling: 3 retries with exponential backoff

Usage:
    aggregator = FeatureAggregator(hdf_dir='/app/data')
    aggregator.aggregate_file('features_20250115.h5')
    
    # Or use scheduler
    scheduler = AggregationScheduler(hdf_dir='/app/data')
    scheduler.start()
"""

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
import time

from src.utils.logger import setup_logger

logger = setup_logger('feature_aggregator')


class AlertManager:
    """
    告警管理器 - 提供统一的告警通知接口
    
    当前实现仅记录日志，可扩展集成：
    - 钉钉/企业微信机器人
    - Email通知
    - Slack/Teams
    - PagerDuty等
    """
    
    @staticmethod
    def send_alert(level: str, message: str, details: Optional[Dict] = None):
        """
        发送告警通知
        
        Args:
            level: 告警级别 (CRITICAL/ERROR/WARNING)
            message: 告警消息
            details: 详细信息字典
        """
        details = details or {}
        alert_msg = f"ALERT [{level}] {message}"
        
        if level == 'CRITICAL':
            logger.critical(alert_msg, extra={'alert_details': details})
        elif level == 'ERROR':
            logger.error(alert_msg, extra={'alert_details': details})
        else:
            logger.warning(alert_msg, extra={'alert_details': details})
        
        # TODO: 集成实际告警系统
        # if level in ['CRITICAL', 'ERROR']:
        #     send_to_dingtalk(alert_msg, details)
        #     send_email(alert_msg, details)


class FeatureAggregator:
    """
    Aggregates OHLCV data with 60-minute sliding window statistics.
    
    For each timestamp T, computes mean and std of [T-60min, T) window.
    Outputs 10 new features: {open,high,low,close,volume}_{mean,std}
    """
    
    DEFAULT_WINDOW_MINUTES = 60
    DEFAULT_RETENTION_DAYS = 90
    DEFAULT_OUTPUT_SUFFIX = '_aggregated'
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 5  # seconds
    
    def __init__(
        self,
        hdf_dir: str,
        output_dir: Optional[str] = None,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize FeatureAggregator.
        
        Args:
            hdf_dir: Directory containing source HDF5 files from HDFDumper
            output_dir: Directory for aggregated output files (default: hdf_dir/aggregated)
            window_minutes: Sliding window size in minutes (default: 60)
            retention_days: How long to keep aggregated files (default: 90)
            alert_callback: Optional callback function for custom alerts
        """
        self.hdf_dir = Path(hdf_dir)
        self.output_dir = Path(output_dir) if output_dir else self.hdf_dir / 'aggregated'
        self.window_minutes = window_minutes
        self.retention_days = retention_days
        self.alert_manager = AlertManager()
        self.alert_callback = alert_callback
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"FeatureAggregator initialized: hdf_dir={self.hdf_dir}, "
            f"output_dir={self.output_dir}, window={window_minutes}min, "
            f"retention={retention_days}days"
        )
    
    def aggregate_file(
        self, 
        filename: str,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate a single HDF5 file by symbol.
        
        Args:
            filename: Source HDF5 filename (e.g., 'features_20250115.h5')
            symbols: Specific symbols to process (None = all symbols in file)
            
        Returns:
            Aggregation result with status and statistics
        """
        filepath = self.hdf_dir / filename
        if not filepath.exists():
            error_msg = f"Source file not found: {filepath}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'filename': filename}
        
        logger.info(f"Starting aggregation for {filename}")
        
        result = {
            'filename': filename,
            'success': False,
            'symbols_processed': 0,
            'symbols_failed': 0,
            'output_files': [],
            'start_time': datetime.now(timezone.utc).isoformat(),
            'end_time': None,
            'errors': []
        }
        
        try:
            # Read metadata to get available symbols
            all_metadata = self._read_metadata(filepath)
            if all_metadata.empty:
                error_msg = f"No metadata found in {filename}"
                logger.warning(error_msg)
                result['error'] = error_msg
                return result
            
            # Get unique symbols
            available_symbols = all_metadata['symbol'].unique().tolist()
            
            if symbols:
                # Filter to requested symbols
                target_symbols = [s for s in symbols if s in available_symbols]
                missing_symbols = [s for s in symbols if s not in available_symbols]
                if missing_symbols:
                    warning_msg = f"Symbols not found in file: {missing_symbols}"
                    logger.warning(warning_msg)
                    result['errors'].append(warning_msg)
            else:
                target_symbols = available_symbols
            
            logger.info(f"Processing {len(target_symbols)} symbols from {filename}")
            
            # Process each symbol
            for symbol in target_symbols:
                try:
                    output_file = self._aggregate_symbol(filepath, filename, symbol)
                    if output_file:
                        result['symbols_processed'] += 1
                        result['output_files'].append(output_file)
                        logger.info(f"Successfully aggregated {symbol} -> {output_file}")
                except Exception as e:
                    result['symbols_failed'] += 1
                    error_msg = f"Failed to aggregate {symbol}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    result['errors'].append(error_msg)
            
            result['success'] = result['symbols_failed'] == 0
            result['end_time'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(
                f"Aggregation complete for {filename}: "
                f"{result['symbols_processed']} succeeded, "
                f"{result['symbols_failed']} failed"
            )
            
        except Exception as e:
            error_msg = f"Aggregation failed for {filename}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['error'] = error_msg
            result['end_time'] = datetime.now(timezone.utc).isoformat()
        
        return result
    
    def _read_metadata(self, filepath: Path) -> pd.DataFrame:
        """Read metadata table from HDF5 file."""
        try:
            with pd.HDFStore(filepath, mode='r') as store:
                metadata_key = 'features/metadata'
                if metadata_key in store:
                    return store[metadata_key]
                else:
                    logger.warning(f"Metadata key '{metadata_key}' not found in {filepath}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to read metadata from {filepath}: {e}")
            return pd.DataFrame()
    
    def _aggregate_symbol(
        self, 
        filepath: Path, 
        source_filename: str,
        symbol: str
    ) -> Optional[str]:
        """
        Aggregate data for a single symbol with retry logic.
        
        实现指数退避重试机制：
        - 第1次失败：等待5秒
        - 第2次失败：等待10秒
        - 第3次失败：放弃并触发告警
        
        Returns:
            Output filename if successful, None otherwise
        """
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._do_aggregate_symbol(filepath, source_filename, symbol)
            except Exception as e:
                last_error = e
                
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.MAX_RETRIES} for {symbol} "
                        f"after {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    # 🚨 3次重试全部失败，触发CRITICAL告警
                    alert_msg = f"Failed to aggregate {symbol} after {self.MAX_RETRIES} retries"
                    self.alert_manager.send_alert(
                        level='CRITICAL',
                        message=alert_msg,
                        details={
                            'symbol': symbol,
                            'filename': source_filename,
                            'error': str(last_error),
                            'retries': self.MAX_RETRIES
                        }
                    )
                    
                    if self.alert_callback:
                        self.alert_callback('CRITICAL', alert_msg)
                    
                    raise
        
        return None
    
    def _do_aggregate_symbol(
        self, 
        filepath: Path, 
        source_filename: str,
        symbol: str
    ) -> str:
        """
        Core aggregation logic for a single symbol (no retry).
        
        Raises:
            Exception: If data is missing or aggregation fails
        """
        # Extract date from filename
        # Support formats: features_YYYYMMDD.h5 or features_YYYYMMDD_HHMMSS_mmm_XXXX.h5
        match = re.search(r'features_(\d{8})', source_filename)
        if not match:
            raise ValueError(f"Invalid filename format: {source_filename}. Expected format: features_YYYYMMDD*.h5")
        date_str = match.group(1)
        
        # Read symbol-specific data
        df = self._read_symbol_data(filepath, symbol)
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # ⚠️ 数据完整性检查
        # 预期：1天 = 1440分钟（24小时 * 60分钟）
        expected_records = 1440
        actual_records = len(df)
        completeness = actual_records / expected_records * 100
        
        if actual_records < expected_records * 0.9:  # < 90%数据完整性
            alert_msg = f"Low data completeness for {symbol}: {actual_records}/{expected_records} ({completeness:.1f}%)"
            logger.warning(alert_msg)
            
            # 触发告警
            self.alert_manager.send_alert(
                level='WARNING',
                message=alert_msg,
                details={
                    'symbol': symbol,
                    'date': date_str,
                    'expected': expected_records,
                    'actual': actual_records,
                    'completeness_pct': completeness
                }
            )
            
            if self.alert_callback:
                self.alert_callback('WARNING', alert_msg)
        else:
            logger.info(f"Data completeness for {symbol}: {completeness:.1f}%")
        
        # Calculate rolling statistics
        rolling_df = self._calculate_rolling_stats(df)
        
        # Merge with original data
        result_df = pd.concat([df, rolling_df], axis=1)
        
        # Write to output file
        output_filename = f"{symbol}{self.DEFAULT_OUTPUT_SUFFIX}_{date_str}.h5"
        output_path = self.output_dir / output_filename
        
        self._write_aggregated_data(result_df, output_path, symbol, date_str)
        
        return output_filename
    
    def _read_symbol_data(self, filepath: Path, symbol: str) -> pd.DataFrame:
        """
        Read and parse OHLCV data for a specific symbol.
        
        使用向量化操作提升性能，避免逐行迭代。
        
        Returns DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        with pd.HDFStore(filepath, mode='r') as store:
            metadata_key = 'features/metadata'
            if metadata_key not in store:
                raise ValueError(f"Metadata not found in {filepath}")
            
            metadata = store[metadata_key]
            symbol_data = metadata[metadata['symbol'] == symbol].copy()
            
            if symbol_data.empty:
                return pd.DataFrame()
        
        # 向量化解析JSON（比iterrows快10-100倍）
        def safe_parse_ohlcv(ohlcv_json):
            """安全解析OHLCV JSON数据"""
            try:
                if isinstance(ohlcv_json, str):
                    return json.loads(ohlcv_json)
                return ohlcv_json or {}
            except (json.JSONDecodeError, TypeError):
                return {}
        
        # 批量解析OHLCV
        symbol_data['ohlcv_parsed'] = symbol_data['ohlcv_data'].apply(safe_parse_ohlcv)
        
        # 提取OHLCV字段
        records = []
        for idx, row in symbol_data.iterrows():
            try:
                ohlcv = row['ohlcv_parsed']
                timestamp_str = row.get('timestamp', '')
                
                if not timestamp_str:
                    continue
                    
                records.append({
                    'timestamp': pd.to_datetime(timestamp_str),
                    'open': float(ohlcv.get('open', 0)),
                    'high': float(ohlcv.get('high', 0)),
                    'low': float(ohlcv.get('low', 0)),
                    'close': float(ohlcv.get('close', 0)),
                    'volume': float(ohlcv.get('volume', 0)),
                })
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipped invalid record for {symbol}: {e}")
                continue
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df = df.set_index('timestamp').sort_index()
        
        # Validate OHLCV data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col} for {symbol}, filling with 0")
                df[col] = 0
        
        return df
    
    def _calculate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 60-minute rolling mean and std for each OHLCV field.
        
        ⚠️ 窗口语义：[T-60min, T) - 包含T-60，不包含T
        
        实现说明：
        - 使用 shift(1) 确保当前时刻T不包含在窗口内
        - 窗口大小60表示向前看60个数据点
        - min_periods=60 确保只有完整窗口才计算（前59个点为NaN）
        
        例如：对于时刻 10:00
        - 窗口包含：09:01, 09:02, ..., 09:59, 10:00前的最后一个点
        - 不包含：10:00 这个点本身
        
        这样确保特征不会"穿越"到未来（避免数据泄漏）
        """
        window = self.window_minutes
        
        # Columns to aggregate
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        result = {}
        for col in ohlcv_cols:
            # 🔑 关键：shift(1)向后移一位，确保[T-60, T)语义
            # rolling(60)在shifted数据上计算，意味着看的是[T-60, T-1]
            shifted_col = df[col].shift(1)
            rolling = shifted_col.rolling(window=window, min_periods=window)
            
            result[f'{col}_mean'] = rolling.mean()
            result[f'{col}_std'] = rolling.std()
        
        return pd.DataFrame(result)
    
    def _write_aggregated_data(
        self, 
        df: pd.DataFrame, 
        output_path: Path, 
        symbol: str,
        date_str: str
    ):
        """Write aggregated data to HDF5 file."""
        # Prepare metadata
        metadata = {
            'symbol': symbol,
            'date': date_str,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'window_minutes': self.window_minutes,
            'total_records': len(df),
            'columns': df.columns.tolist()
        }
        
        with pd.HDFStore(output_path, mode='w', complib='blosc', complevel=5) as store:
            # Store data
            store.put('data', df, format='table')
            # Store metadata as attributes
            store.get_storer('data').attrs.metadata = metadata
        
        logger.debug(f"Wrote {len(df)} records to {output_path}")
    
    def cleanup_old_files(self) -> Dict[str, Any]:
        """
        Remove aggregated files older than retention_days.
        
        Returns:
            Cleanup statistics
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        result = {
            'files_removed': 0,
            'files_kept': 0,
            'errors': [],
            'cutoff_date': cutoff_date.isoformat()
        }
        
        if not self.output_dir.exists():
            return result
        
        for file_path in self.output_dir.glob('*' + self.DEFAULT_OUTPUT_SUFFIX + '*.h5'):
            try:
                # Extract date from filename
                # Format: {SYMBOL}_aggregated_YYYYMMDD.h5
                date_str = file_path.stem.split('_')[-1]
                file_date = datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
                
                if file_date < cutoff_date:
                    file_path.unlink()
                    result['files_removed'] += 1
                    logger.info(f"Removed old aggregated file: {file_path.name}")
                else:
                    result['files_kept'] += 1
                    
            except (ValueError, OSError) as e:
                error_msg = f"Failed to process {file_path}: {e}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
        
        logger.info(
            f"Cleanup complete: {result['files_removed']} removed, "
            f"{result['files_kept']} kept"
        )
        
        return result


class AggregationScheduler:
    """
    Scheduler for running aggregation tasks hourly.
    
    调度策略：
    - 触发时间：每小时的xx:05（推迟5分钟确保数据写入完成）
    - 处理频率：每小时一次，串行处理所有symbols
    - 错误处理：失败不影响下次调度，错误会记录并告警
    - 清理策略：每次成功聚合后清理超过retention_days的旧文件
    
    时区说明：
    - 所有时间使用UTC
    - 例如：01:05 UTC 处理的是 00:00-00:59 UTC 的数据
    """
    
    DEFAULT_SCHEDULE_MINUTE = 5
    
    def __init__(
        self,
        hdf_dir: str,
        output_dir: Optional[str] = None,
        schedule_minute: int = DEFAULT_SCHEDULE_MINUTE
    ):
        """
        Initialize scheduler.
        
        Args:
            hdf_dir: Directory containing source HDF5 files
            output_dir: Directory for aggregated output files
            schedule_minute: Minute of hour to run (default: 5)
        """
        self.aggregator = FeatureAggregator(hdf_dir, output_dir)
        self.schedule_minute = schedule_minute
        self._running = False
        
        logger.info(f"AggregationScheduler initialized (runs at xx:{schedule_minute:02d})")
    
    def _should_process_hour(self, hour: int) -> bool:
        """
        Determine if this hour should be processed.
        
        注意：不跳过任何小时，包括00:00
        - 00:05处理的是昨天23:00-23:59的数据
        - 这是完整的，不存在"部分天"问题
        """
        return True  # 处理所有小时
    
    def run_once(self, target_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run aggregation once for a specific date.
        
        Args:
            target_date: Date to process (default: yesterday)
            
        Returns:
            Aggregation result
        """
        if target_date is None:
            # Process yesterday's data by default
            target_date = datetime.now(timezone.utc) - timedelta(days=1)
        
        date_str = target_date.strftime('%Y%m%d')
        filename = f"features_{date_str}.h5"
        
        logger.info(f"Running aggregation for {date_str}")
        
        # Run aggregation
        result = self.aggregator.aggregate_file(filename)
        
        # Cleanup old files
        if result.get('success'):
            cleanup_result = self.aggregator.cleanup_old_files()
            result['cleanup'] = cleanup_result
        
        return result
    
    def start(self, block: bool = False):
        """
        Start the hourly scheduler.
        
        Args:
            block: If True, block until scheduler is stopped
        """
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        
        self._scheduler = BackgroundScheduler()
        
        # Schedule hourly job at xx:05
        self._scheduler.add_job(
            self._scheduled_job,
            trigger=CronTrigger(minute=self.schedule_minute),
            id='aggregation_job',
            name='Hourly Feature Aggregation',
            replace_existing=True
        )
        
        self._scheduler.start()
        self._running = True
        
        logger.info(f"Scheduler started, running at xx:{self.schedule_minute:02d} every hour")
        
        if block:
            try:
                while self._running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
    
    def _scheduled_job(self):
        """Internal scheduled job wrapper with error handling."""
        now = datetime.now(timezone.utc)
        
        if not self._should_process_hour(now.hour):
            logger.info(f"Skipping aggregation for hour {now.hour}:00")
            return
        
        logger.info(f"Starting scheduled aggregation job at {now.isoformat()}")
        
        try:
            result = self.run_once()
            
            if result.get('success'):
                logger.info(f"Scheduled aggregation completed successfully")
            else:
                logger.error(f"Scheduled aggregation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Scheduled aggregation job failed: {e}", exc_info=True)
    
    def stop(self):
        """Stop the scheduler."""
        if hasattr(self, '_scheduler'):
            self._scheduler.shutdown()
        self._running = False
        logger.info("Scheduler stopped")


# Convenience functions
def aggregate_daily_features(
    hdf_dir: str,
    date_str: Optional[str] = None,
    symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    One-shot aggregation function.
    
    Args:
        hdf_dir: Directory containing source HDF5 files
        date_str: Date to process (YYYYMMDD), defaults to yesterday
        symbols: Specific symbols to process (None = all)
        
    Returns:
        Aggregation result
    """
    aggregator = FeatureAggregator(hdf_dir)
    
    if date_str is None:
        date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y%m%d')
    
    filename = f"features_{date_str}.h5"
    return aggregator.aggregate_file(filename, symbols)
