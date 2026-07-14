import os
import sys
import logging
import uuid
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar

# 全局日志配置
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
# 添加文件名(filename)、函数名(funcName)和行号(lineno)
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(req_id)s] - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ContextVar 用于存储请求 ID（线程/协程安全）
request_id_var: ContextVar[str] = ContextVar('request_id', default='-')


def set_request_id(req_id: str) -> None:
    """设置当前上下文中的请求 ID"""
    normalized = normalize_request_id(req_id)
    request_id_var.set(normalized or '-')


def get_request_id() -> str:
    """获取当前上下文中的请求 ID，如果没有则生成一个新的"""
    req_id = request_id_var.get()
    if req_id == '-':
        req_id = str(uuid.uuid4())[:8]
        request_id_var.set(req_id)
    return req_id


def clear_request_id() -> None:
    """清除当前上下文中的请求 ID"""
    request_id_var.set('-')


def normalize_request_id(value) -> str:
    """标准化 request_id，确保返回可安全使用的字符串。"""
    if value is None:
        return ''
    return str(value).strip()


class RequestIdFormatter(logging.Formatter):
    """自定义格式化器：如果 record 中没有 req_id，则自动从 ContextVar 获取"""
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, 'req_id'):
            # 尝试获取 req_id，如果报错则设为 '-'
            try:
                record.req_id = request_id_var.get()
            except Exception:
                record.req_id = '-'
        return super().format(record)


LOG_DIR = os.environ.get('LOG_DIR', '/app/logs')
LOG_TO_FILE = os.environ.get('LOG_TO_FILE', 'true').lower() == 'true'
LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', str(10 * 1024 * 1024)))  # 10MB
LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', '5'))
LOG_FILE_NAME = os.environ.get('LOG_FILE_NAME', 'server.log')  # 统一日志文件名

# 缓存已创建的logger
_loggers = {}

# 全局共享的 file handler（避免重复创建）
_shared_file_handler = None

# 配置根logger以确保基本日志能够输出
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# 清除已有的 handler 避免重复
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)

# 创建默认的 console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(RequestIdFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
root_logger.addHandler(console_handler)


def _get_or_create_file_handler():
    """获取或创建共享的文件 handler，失败时返回 None"""
    global _shared_file_handler
    
    if _shared_file_handler is None and LOG_TO_FILE:
        try:
            # 确保日志目录存在
            os.makedirs(LOG_DIR, exist_ok=True)
            
            # 创建统一的日志文件路径
            log_file = os.path.join(LOG_DIR, LOG_FILE_NAME)
            
            # 创建格式化器
            formatter = RequestIdFormatter(
                LOG_FORMAT,
                datefmt=LOG_DATE_FORMAT
            )
            
            # 创建 rotating file handler
            _shared_file_handler = RotatingFileHandler(
                log_file,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            _shared_file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
            _shared_file_handler.setFormatter(formatter)
        except (OSError, PermissionError) as e:
            # 如果无法创建日志目录（如只读文件系统），仅输出到控制台
            print(f"Warning: Cannot create log directory {LOG_DIR}: {e}. Logging to console only.", file=sys.stderr)
            _shared_file_handler = None
    
    return _shared_file_handler

def setup_logger(name: str, level: str = None, log_file_name: str = None) -> logging.Logger:
    """
    创建或获取一个配置好的logger
    
    Args:
        name: logger名称,通常使用模块名
        level: 日志级别,默认使用环境变量 LOG_LEVEL
        log_file_name: 自定义日志文件名, 默认使用 LOG_FILE_NAME
        
    Returns:
        配置好的 Logger 实例
    """
    # 如果已经创建过,直接返回
    if name in _loggers:
        return _loggers[name]
    
    # 创建logger
    logger = logging.getLogger(name)

    # 如果 logger 已经有 handler,直接返回,避免重复添加
    if logger.handlers:
        return logger

    # 设置日志级别
    log_level = level or os.environ.get('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # 创建格式化器
    formatter = RequestIdFormatter(
        LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )

    # 创建 console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 添加文件 handler
    if log_file_name:
        # 使用自定义日志文件名
        log_file_path = os.path.join(LOG_DIR, log_file_name)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # 使用默认的共享 handler
        file_handler = _get_or_create_file_handler()
        if file_handler:
            logger.addHandler(file_handler)
    
    # 防止日志向上传播到 root logger
    logger.propagate = False
    
    # 缓存logger
    _loggers[name] = logger
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    获取logger的别名方法
    
    Args:
        name: logger名称
        
    Returns:
        Logger 实例
    """
    return setup_logger(name)
