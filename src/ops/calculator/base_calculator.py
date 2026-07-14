# this is the base class for all the retrievers
from typing import List
import pandas as pd

from ..base_op import BaseOp

class BaseCalculator(BaseOp):
    """
    特征处理基类：
    输入：pandas DataFrame
    输出：嵌套列表 List[List[float]]
    """

    def __init__(self, input_data : pd.DataFrame):
        """
        :param input_data: 输入数据
        """
        super().__init__()
        self.input_data = input_data

    def execute(self) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")
