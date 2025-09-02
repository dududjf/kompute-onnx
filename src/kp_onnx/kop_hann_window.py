import kp
import numpy as np
from .shader_utils import compile_source
from typing import List, Tuple


class HannWindowOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source('''
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer buf_out_tensor { float out_data[]; };
layout (constant_id = 0) const float size = 0;
layout (constant_id = 1) const float periodic = 0;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid < uint(size)) {
        float n = float(gid);
        float N_1 = (periodic > 0.5) ? size : (size - 1.0);
        float pi = 3.14159265359;
        float result = pow(sin(n * pi / N_1), 2.0);
        out_data[gid] = result;
    }
}
''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"HannWindowOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"HannWindowOp({device_name})"

    def run(self, *inputs, dtype=np.float32):
        if len(inputs) == 1:
            inputs = [inputs[0], np.array([1.0], dtype=np.float32)]  # 默认 periodic = True

        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        size = int(input_tensors[0][0].data())

        if size == 0:
            for tensor, _ in input_tensors:
                del tensor
            return [np.array([], dtype=dtype)]

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        if updated_algorithms:
            seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
            for alg in updated_algorithms:
                seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape).astype(dtype)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self,input_tensors: List[Tuple[kp.Tensor, List[int]]],updated_algorithms: List[kp.Algorithm],
        updated_tensors: List[kp.Tensor]) -> List[Tuple[kp.Tensor, List[int]]]:

        tensor_size, shape_size = input_tensors[0]
        tensor_periodic, shape_periodic = input_tensors[1]

        size = int(tensor_size.data())
        periodic = float(tensor_periodic.data())

        if size == 1:
            tensor_out = self.manager.tensor(np.array([0.], dtype=np.float32))
            updated_tensors.append(tensor_out)
            return [(tensor_out, [1])]

        tensor_shape = [size]
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        updated_algorithms.append(self.manager.algorithm([tensor_out],
                                                        self.compiled_shader,
                                                        (size, 1, 1),
                                                        [float(size), periodic],
                                                        []))
        return [(tensor_out, tensor_shape)]
