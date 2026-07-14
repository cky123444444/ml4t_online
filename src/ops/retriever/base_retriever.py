# this is the base class for all the retrievers
from ..base_op import BaseOp

class BaseRetriever(BaseOp):
    """
    请求数据源基类：
    输入：无
    输出：原始数据
    """
    def __init__(self):
        super().__init__()

    def execute(self):
        raise NotImplementedError("Subclasses must implement this method")
