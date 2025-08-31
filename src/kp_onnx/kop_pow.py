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

        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert len(input_tensors) == 2, "PowOp requires two inputs"
        tensor_in = input_tensors[0][0]
        exponent = float(input_tensors[1][0].data())
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