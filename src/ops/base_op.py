# this is the base class for all the ops
class BaseOp:
    def __init__(self):
        pass

    def execute(self):
        raise NotImplementedError("Subclasses must implement this method")