import sqlite3
import os
from src.utils.logger import setup_logger

logger = setup_logger('db')

def get_conn(db_path=None):
    """
    Get SQLite connection with automatic directory creation.
    
    Args:
        db_path: Path to database file. If None, tries multiple env vars or default.
        
    Returns:
        SQLite connection object
    """
    # 如果未提供路径，使用环境变量或默认值
    # 优先级：传入参数 > SCORE_DB_PATH > ORDER_DB_PATH > /app/data/orders.db
    if db_path is None:
        db_path = os.environ.get('SCORE_DB_PATH') or os.environ.get('ORDER_DB_PATH', '/app/data/orders.db')
    
    # ✅ 自动创建目录（参考 hdf_dumper.py 的实现）
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
        except Exception as e:
            logger.error(f"Failed to create database directory {db_dir}: {e}", exc_info=True)
            raise
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    
    logger.debug(f"Connected to database: {db_path}")
    return conn
