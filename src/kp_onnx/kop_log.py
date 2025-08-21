import kp
import numpy as np
from pyshader import python2shader, ivec2, f32, Array
from pyshader.stdlib import log


@python2shader
def compute_shader_log(index=("input", "GlobalInvocationId", ivec2),
                       in_data=("buffer", 0, Array(f32)),
                       out_data=("buffer", 1, Array(f32))):
    i = index.x
    out_data[i] = log(in_data[i])


class LogOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output
        self.shader = compute_shader_log.to_spirv()

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"LogOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"LogOp({device_name})"

    def run(self, *inputs):
        data = inputs[0].astype(np.float32)
        flat_data = data.reshape(-1)
        tensor_in = self.manager.tensor(flat_data)
        tensor_out = self.manager.tensor(np.zeros_like(flat_data))

        algo = self.manager.algorithm([tensor_in, tensor_out], self.shader)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        outputs = [tensor_out.data().reshape(data.shape)]
        del tensor_in, tensor_out
        return outputs
