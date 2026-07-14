# this is the base class for all the adaptors
from ..base_op import BaseOp
import pandas as pd

class BaseAdaptor(BaseOp):
    """
    适配器基类：
    输入：原始数据
    输出：pandas DataFrame
    """

    def __init__(self, input_data):
        self.input_data = input_data

    def execute(self) -> pd.DataFrame:
        # 输出 df
        raise NotImplementedError("Subclasses must implement this method")
