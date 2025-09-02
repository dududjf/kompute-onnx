import kp
import numpy as np
from .shader_utils import compile_source


class ReshapeOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source('''
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer buf_in_tensor { float in_data[]; };
layout(set=0, binding=1) buffer buf_out_tensor { float out_data[]; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    out_data[gid] = in_data[gid];
}
''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) >= 2, "ReshapeOp requires at least 2 inputs"

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

    def fuse(self,input_tensors: list[tuple[kp.Tensor, list[int]]],updated_algorithms: list[kp.Algorithm],
        updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert len(input_tensors) >= 2, "ReshapeOp requires at least 2 inputs"

        tensor_in = input_tensors[0][0]
        tensor_shape = input_tensors[0][1]

        shape_data = input_tensors[1][0].data()
        new_shape = [int(x) for x in shape_data]

        minus_one_count = new_shape.count(-1)
        if minus_one_count == 1:
            idx = new_shape.index(-1)
            new_shape[idx] = int(np.prod(tensor_shape) / np.prod([d for d in new_shape if d != -1]))
        elif minus_one_count > 1:
            raise ValueError("Only one dimension can be -1")

        allowzero = 0
        if len(input_tensors) > 2:
            allowzero = int(input_tensors[2][0].data())
        if allowzero == 0:
            for idx, dim in enumerate(new_shape):
                if dim == 0:
                    new_shape[idx] = tensor_shape[idx]

        assert np.prod(new_shape) == np.prod(tensor_shape), \
            f"Cannot reshape tensor of size {np.prod(tensor_shape)} into shape {new_shape}"

        size = np.prod(new_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)
        updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out],
                                                         self.compiled_shader,
                                                         (size, 1, 1),
                                                         [],
                                                         []))
        return [(tensor_out, new_shape)]
