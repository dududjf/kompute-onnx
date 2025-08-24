import kp
import numpy as np
from .shader_utils import compile_source


class PowOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source('''
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer buf_in_tensor  { float in_data[]; };
layout(set=0, binding=1) buffer buf_out_tensor { float out_data[];};
layout (constant_id = 0) const float exponent = 0;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    float x = in_data[gid];
    out_data[gid] = pow(x, exponent);
}
''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"PowOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"PowOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "PowOp requires two inputs"
        tensor_shape = inputs[0].shape
        numpy_in = inputs[0].reshape(-1).astype(np.float32)
        exponent = float(inputs[1])
        tensor_in = self.manager.tensor(numpy_in)
        tensor_out = self.manager.tensor(np.zeros_like(numpy_in))
        algo = self.manager.algorithm([tensor_in, tensor_out],
                                      self.compiled_shader,
                                      (len(numpy_in), 1, 1),
                                      [exponent],
                                      [])
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
        assert len(input_tensors) == 2, "PowOp requires two inputs"
        tensor_in = input_tensors[0][0]
        exponent = float(input_tensors[1][0])
        tensor_shape = input_tensors[0][1]
        size = np.prod(tensor_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)
        updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out],
                                                         self.compiled_shader,
                                                         (size, 1, 1),
                                                         [exponent],
                                                         []))
        return [(tensor_out, tensor_shape)]
