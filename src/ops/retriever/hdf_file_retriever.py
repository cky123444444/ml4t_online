import pandas as pd
import os
from .base_retriever import BaseRetriever
from src.utils.logger import setup_logger

logger = setup_logger('hdf_file_retriever')

class HDFFileRetriever(BaseRetriever):
    """
    从本地HDF文件读取DataFrame的检索器
    """
    def __init__(self, hdf_path, key):
        """
        :param hdf_path: HDF文件路径
        :param key: HDF文件中的key
        """
        super().__init__()
        self.hdf_path = hdf_path
        self.key = key

    def execute(self):
        """
        读取HDF文件中的DataFrame
        :return: pandas.DataFrame
        """
        if not os.path.exists(self.hdf_path):
            raise FileNotFoundError(f"HDF文件不存在: {self.hdf_path}")
        try:
            df = pd.read_hdf(self.hdf_path, key=self.key)
            return df
        except Exception as e:
            raise RuntimeError(f"读取HDF文件失败: {e}")