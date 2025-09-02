import kp
import numpy as np
from .shader_utils import compile_source


class ConstantOfShapeOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source('''
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Output buffer
layout(set=0, binding=0) buffer buf_out_tensor { float out_data[]; };

// Specialization constants
layout(constant_id = 0) const int dtype = 0; // 0=float32, 1=float64, 2=int32, 3=int64
layout(constant_id = 1) const float value_f = 0.0;
layout(constant_id = 2) const int value_i = 0;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (dtype == 0) { out_data[gid] = value_f; }
    else if (dtype == 1) { out_data[gid] = float(value_f); }
    else if (dtype == 2) { out_data[gid] = float(value_i); }
    else if (dtype == 3) { out_data[gid] = float(value_i); }
}
''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ConstantOfShapeOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ConstantOfShapeOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) >= 1, "ConstantOfShapeOp requires at least one input (shape tensor)"

        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape, out_dtype = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape).astype(out_dtype)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self,input_tensors: list[tuple[kp.Tensor, list[int]]],updated_algorithms: list[kp.Algorithm],
            updated_tensors: list[kp.Tensor]) -> tuple[list[tuple[kp.Tensor, list[int]]], type]:
        shape_tensor, shape_list = input_tensors[0]
        shape_data = shape_tensor.data()
        output_shape = [int(x) for x in shape_data]
        size = int(np.prod(output_shape))

        if len(input_tensors) > 1:
            val_tensor, _ = input_tensors[1]
            val_data = val_tensor.data()
            constant_value = val_data[0] if len(val_data) == 1 else val_data
        else:
            constant_value = 0.0

        if isinstance(constant_value, int):
            if -2147483648 <= constant_value <= 2147483647:
                out_dtype = np.int32
                dtype_code = 2
            else:
                out_dtype = np.int64
                dtype_code = 3
        else:
            out_dtype = np.float32
            dtype_code = 0

        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        if out_dtype in [np.float32, np.float64]:
            spec_consts = [dtype_code, float(constant_value), 0.0]
        else:
            spec_consts = [dtype_code, 0.0, int(constant_value)]

        updated_algorithms.append(self.manager.algorithm([tensor_out],
                                                        self.compiled_shader,
                                                        (size, 1, 1),
                                                        spec_consts,
                                                         []))

        return [(tensor_out, output_shape)], out_dtype

