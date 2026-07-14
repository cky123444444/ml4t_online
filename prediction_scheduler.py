#!/usr/bin/env python3
"""
Prediction Scheduler Service

每隔指定间隔（默认 1 分钟）自动调用 Main Server 的 /predict 端点。
支持通过环境变量配置调用频率、method、重试策略等。

Usage:
    python prediction_scheduler.py

Environment Variables:
    SERVER_URL: Main server URL (default: http://127.0.0.1:8000)
    SCHEDULER_INTERVAL_MINUTES: Interval in minutes (default: 1)
    SCHEDULER_METHOD: Process method - process_1|2|3 (default: process_3)
    SCHEDULER_RETRY_COUNT: Retry count on failure (default: 1)
    LOG_LEVEL: Log level - DEBUG|INFO|WARNING|ERROR (default: INFO)
"""

import os
import sys
import json
import time
import signal
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import setup_logger, set_request_id

# ===========================
# Configuration
# ===========================
SERVER_URL = os.environ.get('SERVER_URL', 'http://127.0.0.1:8000')
SCHEDULER_INTERVAL_MINUTES = int(os.environ.get('SCHEDULER_INTERVAL_MINUTES', '1'))
SCHEDULER_METHOD = os.environ.get('SCHEDULER_METHOD', 'process_3')
SCHEDULER_RETRY_COUNT = int(os.environ.get('SCHEDULER_RETRY_COUNT', '1'))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
REQUEST_TIMEOUT = float(os.environ.get('TS_TIMEOUT', '30'))

# Initialize logger
logger = setup_logger('prediction_scheduler')
logger.setLevel(getattr(logging, LOG_LEVEL))

# ===========================
# Global State
# ===========================
scheduler = None
job_stats = {
    'total_calls': 0,
    'successful_calls': 0,
    'failed_calls': 0,
    'total_retries': 0,
}


def make_predict_request(
    server_url: str,
    method: str,
    timeout: float = REQUEST_TIMEOUT,
    request_id: Optional[str] = None
) -> tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Make a POST request to server's /predict endpoint.
    
    Args:
        server_url: Base URL of the server (e.g., http://127.0.0.1:8000)
        method: Process method (process_1, process_2, process_3)
        timeout: Request timeout in seconds
        request_id: Optional request ID for tracking
    
    Returns:
        Tuple of (success: bool, response_dict: dict, message: str)
    """
    try:
        url = f"{server_url}/predict"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "ple",
            "method": method,
            "req_debug_mode": False,  # Disable debug mode in scheduler for cleaner logs
        }
        if request_id:
            payload['request_id'] = request_id
        
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed_time = time.time() - start_time
        
        # Check HTTP status
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            return False, None, error_msg
        
        # Parse JSON response
        result = response.json()
        msg = f"Success (took {elapsed_time:.2f}s)"
        return True, result, msg
        
    except requests.exceptions.Timeout:
        return False, None, f"Timeout after {timeout}s"
    except requests.exceptions.ConnectionError as e:
        return False, None, f"Connection error: {str(e)[:100]}"
    except requests.exceptions.RequestException as e:
        return False, None, f"Request error: {str(e)[:100]}"
    except json.JSONDecodeError:
        return False, None, "Invalid JSON response"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)[:100]}"


def predict_job():
    """
    Scheduled job that calls the predict endpoint.
    Implements automatic retry on failure.
    """
    import uuid
    request_id = str(uuid.uuid4())[:8]

    # 设置 request_id 到日志上下文，使本服务的日志也能显示 request_id
    set_request_id(request_id)

    job_stats['total_calls'] += 1

    logger.info(f"[Job #{job_stats['total_calls']}] Starting predict request (method={SCHEDULER_METHOD})")
    
    # Try initial call
    success, result, message = make_predict_request(
        SERVER_URL,
        SCHEDULER_METHOD,
        timeout=REQUEST_TIMEOUT,
        request_id=request_id
    )
    
    if success:
        job_stats['successful_calls'] += 1
        # Extract useful info from result
        if isinstance(result, dict):
            model_output = result.get('result', {})
            if isinstance(model_output, dict):
                output_str = json.dumps(model_output, indent=2)
                logger.info(
                    f"[Job #{job_stats['total_calls']}] ✅ Predict successful "
                    f"request_id={request_id}\nOutput:\n{output_str},msg={message}"
                )
            else:
                logger.info(
                    f"[Job #{job_stats['total_calls']}] ✅ Predict successful "
                    f"request_id={request_id}, msg={message}"
                )
        return
    
    # First attempt failed, retry once
    logger.warning(
        f"[Job #{job_stats['total_calls']}] First attempt failed: {message}. Retrying..."
    )
    job_stats['total_retries'] += 1
    
    time.sleep(1)  # Wait 1 second before retry
    
    success_retry, result_retry, message_retry = make_predict_request(
        SERVER_URL,
        SCHEDULER_METHOD,
        timeout=REQUEST_TIMEOUT,
        request_id=request_id
    )
    
    if success_retry:
        job_stats['successful_calls'] += 1
        logger.info(
            f"[Job #{job_stats['total_calls']}] ✅ Predict successful on retry "
            f"request_id={request_id}, msg={message_retry}"
        )
        return
    
    # Both attempts failed
    job_stats['failed_calls'] += 1
    logger.error(
        f"[Job #{job_stats['total_calls']}] ❌ Predict failed after retry. "
        f"request_id={request_id}, error={message_retry}"
    )


def print_stats():
    """Print job statistics."""
    total = job_stats['total_calls']
    if total == 0:
        logger.info("No jobs executed yet")
        return
    
    success_rate = (job_stats['successful_calls'] / total * 100) if total > 0 else 0
    logger.info(
        f"📊 Scheduler Stats: "
        f"total_calls={total}, "
        f"successful={job_stats['successful_calls']}, "
        f"failed={job_stats['failed_calls']}, "
        f"total_retries={job_stats['total_retries']}, "
        f"success_rate={success_rate:.1f}%"
    )


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info("🛑 Received shutdown signal, stopping scheduler...")
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=True)
    print_stats()
    logger.info("Scheduler stopped. Goodbye!")
    sys.exit(0)


def main():
    """Main entry point for the prediction scheduler."""
    global scheduler
    
    logger.info("=" * 70)
    logger.info("🚀 Prediction Scheduler Starting")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  - Server URL: {SERVER_URL}")
    logger.info(f"  - Interval: Every {SCHEDULER_INTERVAL_MINUTES} minute(s)")
    logger.info(f"  - Method: {SCHEDULER_METHOD}")
    logger.info(f"  - Retry count on failure: {SCHEDULER_RETRY_COUNT}")
    logger.info(f"  - Request timeout: {REQUEST_TIMEOUT}s")
    logger.info(f"  - Log level: {LOG_LEVEL}")
    logger.info("=" * 70)
    
    # Validate server URL
    try:
        logger.info("Checking server health...")
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Server health check passed")
        else:
            logger.warning(f"⚠️  Server health check returned {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️  Could not reach server: {e}. Continuing anyway...")
    
    # Initialize scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        predict_job,
        trigger=CronTrigger(minute=f'*/{SCHEDULER_INTERVAL_MINUTES}'),  # 每 N 分钟的整数时刻触发
        id='predict_job',
        name='Predict Job',
        replace_existing=True
    )
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info(f"Starting scheduler with {SCHEDULER_INTERVAL_MINUTES} minute interval...")
        scheduler.start()
        logger.info("✅ Scheduler started successfully")
        
        # Keep the main thread alive
        while True:
            time.sleep(60)
            # Optionally print stats every 60 seconds
            # print_stats()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        signal_handler(None, None)
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)
        print_stats()
        sys.exit(1)


if __name__ == '__main__':
    main()
