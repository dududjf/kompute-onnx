import kp
import numpy as np
from pyshader import python2shader, ivec2, f32, Array
from pyshader.stdlib import cos


@python2shader
def compute_shader_cos(index=("input", "GlobalInvocationId", ivec2),
                       in_data=("buffer", 0, Array(f32)),
                       out_data=("buffer", 1, Array(f32))):
    i = index.x
    out_data[i] = cos(in_data[i])


_cos_code = compute_shader_cos.to_spirv()


class CosOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"CosOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"CosOp({device_name})"

    def run(self, *inputs):
        tensor_shape = inputs[0].shape
        numpy_in = inputs[0].reshape(-1).astype(np.float32)
        tensor_in = self.manager.tensor(numpy_in)
        tensor_out = self.manager.tensor(np.zeros_like(numpy_in))
        algo = self.manager.algorithm([tensor_in, tensor_out], _cos_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        outputs = [tensor_out.data().reshape(tensor_shape)]
        del tensor_in
        del tensor_out
        return outputs
