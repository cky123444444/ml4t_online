"""
Debug utilities for saving intermediate data during development and testing
"""
import os
import json
import pandas as pd
from pathlib import Path
from typing import Union, Any
from src.utils.logger import setup_logger

logger = setup_logger('debug_utils')


def save_debug_data(
    data: Any,
    filename: str,
    output_dir: str = "/app/debug_output",
    data_type: str = "auto"
) -> bool:
    """
    保存调试数据到文件
    
    Args:
        data: 要保存的数据（DataFrame、list、dict 或其他）
        filename: 文件名（可含或不含扩展名）
        output_dir: 输出目录路径（建议使用绝对路径）
        data_type: 数据类型 ("csv", "json", "raw", "auto")
                  "auto" 会根据数据类型自动判断
    
    Returns:
        bool: 保存是否成功
    """
    try:
        # 转换为绝对路径
        output_dir = os.path.abspath(output_dir)
        
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"输出目录: {output_dir}")
        
        # 自动检测数据类型
        if data_type == "auto":
            if isinstance(data, pd.DataFrame):
                data_type = "csv"
            elif isinstance(data, (list, dict)):
                data_type = "json"
            else:
                data_type = "raw"
        
        # 确保文件名有正确的扩展名
        if not any(filename.endswith(ext) for ext in ['.csv', '.json', '.txt', '.log']):
            filename = f"{filename}.{data_type}"
        
        filepath = os.path.join(output_dir, filename)
        
        # 根据数据类型保存
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
            logger.debug(f"DataFrame 已保存至: {filepath}")
        elif data_type == "json" or isinstance(data, (list, dict)):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"JSON 数据已保存至: {filepath}")
        else:
            # 默认保存为文本
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(data))
            logger.debug(f"原始数据已保存至: {filepath}")
        
        # 验证文件是否真的存在
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            logger.debug(f"文件已确认存在，大小: {file_size} bytes")
            return True
        else:
            logger.warning(f"文件保存后不存在: {filepath}")
            return False
        
    except Exception as e:
        logger.error(f"保存数据失败: {e}", exc_info=True)
        return False
