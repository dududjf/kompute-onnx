import numpy as np
import kp
from .shader_utils import compile_source, broadcast_to


class LpNormalizationOp:
    def __init__(self, manager: kp.Manager, axis=-1, p=2):
        self.manager = manager
        self.axis = axis
        self.p = p
        self.compiled_shader_sum = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;

layout (binding = 0) buffer buf_in_tensor  { float in_tensor[];  };
layout (binding = 1) buffer buf_out_tensor { float out_tensor[]; };

layout (constant_id = 0) const float dimension_f = 0;
layout (constant_id = 1) const float block_size_f = 0;
layout (constant_id = 2) const float p_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    
    uint dimension = uint(dimension_f);
    uint block_size = uint(block_size_f);
    
    uint base = gx * dimension * block_size + gy;
    uint out_off = gx * block_size + gy;
    
    float acc = 0.0;
    for (uint i = 0; i < dimension; ++i, base += block_size) {
        acc += pow(in_tensor[base], p_f);
    }

    out_tensor[out_off] = acc;
    
    out_tensor[gx] = pow(out_tensor[gx], 1.0 / p_f);
    
}
""")
        self.compiled_shader_div = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1) in;

layout (binding = 0) buffer buf_in_tensor_1  { float in_tensor_1[];  };
layout (binding = 1) buffer buf_in_tensor_2 { float in_tensor_2[]; };
layout (binding = 2) buffer buf_out_tensor { float out_tensor[]; };

layout (constant_id = 0) const float dimension_f = 0;
layout (constant_id = 1) const float block_size_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    
    uint dimension = uint(dimension_f);
    uint block_size = uint(block_size_f);
    
    // TODO: in_tensor_1 / in_tensor_2
}
""")

    def __repr__(self):
        return f"LpNormalizationOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]]))
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
        tensor_in, shape_in = input_tensors[0]
        axis = self.axis
        p = self.p

        axis += len(shape_in) if axis < 0 else 0

        dimension = shape_in[axis]
        group_x = int(np.prod(shape_in[:axis])) if axis >= 0 else 1
        block_size = int(np.prod(shape_in[axis + 1:])) if axis + 1 < len(shape_in) else 1

        size_out = int(np.prod(shape_in))
        tensor_out_power = self.manager.tensor(np.zeros(size_out, dtype=np.float32))
        tensor_out_sum = self.manager.tensor(np.zeros(group_x * block_size, dtype=np.float32))
        tensor_out = self.manager.tensor(np.zeros(size_out, dtype=np.float32))
        updated_tensors.extend([tensor_out_power, tensor_out_sum, tensor_out])

        updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out_sum],
                                                         self.compiled_shader_sum, (max(group_x, size_out), block_size, 1),
                                                         [dimension, block_size, p],
                                                         []))
        output_shape = shape_in[:axis] + shape_in[axis + 1:]

        updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out_sum, tensor_out],
                                                         self.compiled_shader_div, (group_x, block_size, 1),
                                                         [dimension, block_size],
                                                         []))

        return [(tensor_out, output_shape)]
