import kp
import numpy as np
from pyshader import python2shader, ivec2, f32, Array
from pyshader.stdlib import exp, min, abs

@python2shader
def compute_shader_sigmoid(index=("input", "GlobalInvocationId", ivec2),
                           in_data=("buffer", 0, Array(f32)),
                           out_data=("buffer", 1, Array(f32))):
    i = index.x
    x = in_data[i]
    absx = abs(x)
    neg_absx = -absx
    out_data[i] = exp(min(x, 0.0)) / (1.0 + exp(neg_absx))


_sigmoid_code = compute_shader_sigmoid.to_spirv()


class SigmoidOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"SigmoidOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"SigmoidOp({device_name})"

    def run(self, *inputs):
        tensor_shape = inputs[0].shape
        numpy_in = inputs[0].reshape(-1).astype(np.float32)
        tensor_in = self.manager.tensor(numpy_in)
        tensor_out = self.manager.tensor(np.zeros_like(numpy_in))
        algo = self.manager.algorithm([tensor_in, tensor_out], _sigmoid_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        outputs = [tensor_out.data().reshape(tensor_shape)]
        del tensor_in
        del tensor_out
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in = input_tensors[0][0]
        tensor_shape = input_tensors[0][1]
        size = np.prod(tensor_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)
        updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out], _sigmoid_code))
        return [(tensor_out, tensor_shape)]
