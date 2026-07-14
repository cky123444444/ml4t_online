# this is the base class for all the retrievers
from typing import Dict, List

from ..base_op import BaseOp

class BaseStrategy(BaseOp):
    """
    特征处理基类：
    输入：pandas DataFrame
    输出：嵌套列表 List[List[float]]
    """

    def __init__(self, input_data : Dict[str, List[float]]):
        """
        :param input_data: 模型的输出, Dict[str, List[float]]
        - key 为以下字段:
            "ctrl_return_1h_out",
            "trmt_return_1h_out",
            "ctrl_sigma_1h_out",
            "trmt_sigma_1h_out",
            "t_out",
            "return_1h_eps_out",
            "sigma_1h_eps_out"
        - value 为对应的特征数据, 长度为 batch size
        """
        super().__init__()
        self.input_data = input_data

    def execute(self):
        raise NotImplementedError("Subclasses must implement this method")
