import numpy as np
import kp


class IdentityOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

    def __repr__(self):
        return f"IdentityOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        assert len(inputs) >= 1, "IdentityOp needs the input tensor"
        data = inputs[0].astype(np.float32, copy=True)
        outputs = [data]
        return outputs
