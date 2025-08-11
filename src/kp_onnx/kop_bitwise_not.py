import kp
import numpy as np
from pyshader import python2shader, ivec2, f32, Array


@python2shader
def compute_shader_bitwise_not(index=("input", "GlobalInvocationId", ivec2),
                              in_data=("buffer", 0, Array(f32)),
                              out_data=("buffer", 1, Array(f32))):
    i = index.x
    #~x = -x - 1
    out_data[i] = -in_data[i] - 1.0

_bitwise_not_code = compute_shader_bitwise_not.to_spirv()


class BitwiseNotOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def run(self, *inputs):
        tensor_shape = inputs[0].shape
        numpy_in = inputs[0].reshape(-1).astype(np.float32)
        tensor_in = self.manager.tensor(numpy_in)
        tensor_out = self.manager.tensor(np.zeros_like(numpy_in, dtype=np.float32))
        algo = self.manager.algorithm([tensor_in, tensor_out], _bitwise_not_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        outputs = [tensor_out.data().reshape(tensor_shape).astype(np.int32)] #最后输出转换回 int32
        del tensor_in
        del tensor_out
        return outputs