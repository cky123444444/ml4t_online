#!/usr/bin/env python3
"""
Feature Aggregation Service Entry Point

Runs hourly aggregation job for OHLCV data.
Usage:
    python run_aggregator.py              # Start scheduler (runs at xx:05)
    python run_aggregator.py --once       # Run once for yesterday
    python run_aggregator.py --date 20250115  # Run once for specific date
    python run_aggregator.py --symbols BTCUSDT ETHUSDT  # Process specific symbols
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ops.aggregator.feature_aggregator import AggregationScheduler, aggregate_daily_features
from src.utils.logger import setup_logger, set_request_id

logger = setup_logger('run_aggregator')


def main():
    # 设置一个默认的 request_id 用于标识 aggregator 服务
    set_request_id('aggregator')
    parser = argparse.ArgumentParser(description='Feature Aggregation Service')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--date', type=str, help='Specific date to process (YYYYMMDD)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to process')
    parser.add_argument('--hdf-dir', default='/app/data/hdf_dumper', help='HDF input directory')
    parser.add_argument('--output-dir', default='/app/data/hdf_dumper/aggregated', help='Output directory')
    parser.add_argument('--schedule-minute', type=int, default=5, help='Minute of hour to run (default: 5)')
    parser.add_argument('--retention-days', type=int, default=90, help='Retention period for aggregated files')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(args.hdf_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.once:
        # Run once mode
        logger.info(f"Running aggregation once (date={args.date or 'yesterday'})")
        result = aggregate_daily_features(
            hdf_dir=args.hdf_dir,
            date_str=args.date,
            symbols=args.symbols
        )
        
        if result.get('success'):
            logger.info(f"Aggregation completed: {result['symbols_processed']} symbols processed")
            # Run cleanup
            from src.ops.aggregator.feature_aggregator import FeatureAggregator
            aggregator = FeatureAggregator(args.hdf_dir, args.output_dir, retention_days=args.retention_days)
            cleanup = aggregator.cleanup_old_files()
            logger.info(f"Cleanup: {cleanup['files_removed']} files removed, {cleanup['files_kept']} kept")
            return 0
        else:
            logger.error(f"Aggregation failed: {result.get('error')}")
            return 1
    else:
        # Schedule mode
        logger.info("Starting aggregation scheduler...")
        scheduler = AggregationScheduler(
            hdf_dir=args.hdf_dir,
            output_dir=args.output_dir,
            schedule_minute=args.schedule_minute
        )
        
        try:
            scheduler.start(block=True)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            scheduler.stop()
        
        return 0


if __name__ == '__main__':
    sys.exit(main())
