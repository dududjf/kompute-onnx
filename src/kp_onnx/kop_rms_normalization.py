import numpy as np
import kp
from .shader_utils import compile_source, broadcast_to


class RmsNormalizationOp:
    def __init__(self, manager: kp.Manager, axis: int = -1, epsilon: float = 1e-05, stash_type: int = 1):
        self.manager = manager
        self.axis = axis
        self.epsilon = epsilon
        self.stash_type = stash_type
        self.compiled_shader_power = compile_source('''
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

layout(binding=0) readonly  buffer buf_in_tensor  { float in_tensor[];  };
layout(binding=1) writeonly  buffer buf_out_tensor   { float out_tensor[];   };

void main() {
    uint gx = gl_GlobalInvocationID.x;
    out_tensor[gx] = pow(in_tensor[gx], 2);
}
''')
        self.compiled_shader_mean = compile_source('''
#version 450
layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0) buffer buf_in_tensor { float in_tensor[]; };   // 输入张量
layout (binding = 1) buffer buf_out_tensor { float out_tensor[]; }; // 输出张量
layout (constant_id = 0) const float dimension_f = 0;   // 归约维度的大小
layout (constant_id = 1) const float block_size_f = 0;  // 后缀块的大小

void main()
{
    uint gx = gl_GlobalInvocationID.x;  // 块组索引
    uint gy = gl_GlobalInvocationID.y;  // 归约维度内的索引

    uint dimension = uint(dimension_f);
    uint block_size = uint(block_size_f);

    // 计算输入和中间张量的偏移量
    uint in_offset = gx * dimension * block_size + gy;
    uint out_offset = gx * block_size + gy;

    // 累加归约维度上的元素并计算平均值
    out_tensor[out_offset] = 0.0;
    for(uint i = 0; i < dimension; ++i, in_offset += block_size)
        out_tensor[out_offset] += in_tensor[in_offset];
    out_tensor[out_offset] /= dimension;
}
''')
        self.compiled_shader_sq = compile_source('''
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

layout(binding = 0) readonly  buffer buf_in_tensor  { float in_tensor[];  };
layout(binding = 1) writeonly buffer buf_out_tensor { float out_tensor[]; };
layout(constant_id = 0) const float epsilon_f = 0;

void main() {
    uint gx = gl_GlobalInvocationID.x;
    out_tensor[gx] = 1.0 / sqrt(in_tensor[gx] + epsilon_f);
}
''')
        self.compiled_shader_matmul = compile_source('''
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) buffer buf_in_tensor_1 { float in_tensor_1[]; };
layout (binding = 1) buffer buf_in_tensor_2 { float in_tensor_2[]; };
layout (binding = 2) buffer buf_in_tensor_3 { float in_tensor_3[]; };
layout (binding = 3) buffer buf_out_tensor  { float out_tensor[]; };
layout (constant_id = 0) const float size_x_inf = 0;
layout (constant_id = 1) const float size_y_inf = 0;
layout (constant_id = 2) const float size_z_inf = 0;
layout (constant_id = 3) const float size_x_rmsf = 0;
layout (constant_id = 4) const float size_y_rmsf = 0;
layout (constant_id = 5) const float size_z_rmsf = 0;
layout (constant_id = 6) const float size_x_scalef = 0;
layout (constant_id = 7) const float size_y_scalef = 0;
layout (constant_id = 8) const float size_z_scalef = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;
    uint size_x_in = uint(size_x_inf);
    uint size_y_in = uint(size_y_inf);
    uint size_z_in = uint(size_z_inf);
    uint size_x_rms = uint(size_x_rmsf);
    uint size_y_rms = uint(size_y_rmsf);
    uint size_z_rms = uint(size_z_rmsf);
    uint size_x_scale = uint(size_x_scalef);
    uint size_y_scale = uint(size_y_scalef);
    uint size_z_scale = uint(size_z_scalef);
    uint stride_y_in = size_z_in;
    uint stride_x_in = size_y_in * stride_y_in;
    uint stride_y_rms = size_z_rms;
    uint stride_x_rms = size_y_rms * stride_y_rms;
    uint stride_y_scale = size_z_scale;
    uint stride_x_scale = size_y_scale * stride_y_scale;
    uint stride_y = size_z_in;
    uint stride_x = size_y_in * stride_y;
    uint x_in = min(gx, size_x_in - 1);
    uint y_in = min(gy, size_y_in - 1);
    uint z_in = min(gz, size_z_in - 1);
    uint x_rms = min(gx, size_x_rms - 1);
    uint y_rms = min(gy, size_y_rms - 1);
    uint z_rms = min(gz, size_z_rms - 1);
    uint x_scale = min(gx, size_x_scale - 1);
    uint y_scale = min(gy, size_y_scale - 1);
    uint z_scale = min(gz, size_z_scale - 1);
    uint p_in = x_in * stride_x_in + y_in * stride_y_in + z_in;
    uint p_rms = x_rms * stride_x_rms + y_rms * stride_y_rms + z_rms;
    uint p_scale = x_scale * stride_x_scale + y_scale * stride_y_scale + z_scale;
    out_tensor[gx * stride_x + gy * stride_y + gz] = in_tensor_1[p_in] * in_tensor_2[p_rms] * in_tensor_3[p_scale];
}''')

    def __repr__(self):
        return f"RmsNormalizationOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_in = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(shape_in)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        tensor_scale, shape_scale = input_tensors[1]

        assert self.stash_type == 1, "RMSNormalization not implemented for stash_type != 1."
        epsilon = self.epsilon

        axis = self.axis if self.axis >= 0 else len(shape_in) + self.axis

        axes = list(range(axis, len(shape_in)))
        axis_present = [False] * len(shape_in)
        for axis in axes:
            axis_present[axis] = True

        tensor_pow = self.manager.tensor(np.zeros(np.prod(shape_in), dtype=np.float32))
        updated_tensors.append(tensor_pow)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_pow],
            self.compiled_shader_power
        ))

        tensor_mean = tensor_pow
        block_size = 1
        for i in reversed(range(len(shape_in))):
            if axis_present[i] and shape_in[i] > 1:
                group_x = int(np.prod(shape_in[:i])) if i >= 0 else 1
                workgroup = (group_x, block_size, 1)
                numpy_out = np.zeros(group_x * block_size, dtype=np.float32)
                tensor_in = tensor_mean
                tensor_mean = self.manager.tensor(numpy_out)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_mean],
                    self.compiled_shader_mean,
                    workgroup,
                    [shape_in[i], block_size],
                    []
                ))
                updated_tensors.append(tensor_mean)
            else:
                block_size *= int(shape_in[i])

        shape_rms = [1 if axis_present[i] else shape_in[i] for i in range(len(shape_in))]
        tensor_rms = self.manager.tensor(np.zeros(np.prod(shape_rms), dtype=np.float32))
        updated_algorithms.append(self.manager.algorithm(
            [tensor_mean, tensor_rms],
            self.compiled_shader_sq,
            (int(np.prod(shape_rms)), 1, 1),
            spec_consts=[epsilon]
        ))

        max_dims = max(len(shape_in), len(shape_rms), len(shape_scale))

        new_shape_rms = [1] * (max_dims - len(shape_rms)) + shape_rms
        new_shape_scale = [1] * (max_dims - len(shape_scale)) + shape_scale

        new_rms = tensor_rms
        algorithms_rms, next_tensors_rms = [], []
        if shape_in[:-2] != new_shape_rms[:-2] and not all(e == 1 for e in new_shape_rms[:-2]):
            final_shape_rms = shape_in[:-2] + list(new_shape_rms[-2:])
            new_rms = broadcast_to(tensor_rms, new_shape_rms, final_shape_rms, algorithms_rms, next_tensors_rms, self.manager)
            updated_algorithms.extend(algorithms_rms)
            new_shape_rms = final_shape_rms

        new_scale = tensor_scale
        algorithms_scale, next_tensors_scale = [], []
        if shape_in[:-2] != new_shape_scale[:-2] and not all(e == 1 for e in new_shape_scale[:-2]):
            final_shape_scale = shape_in[:-2] + list(new_shape_scale[-2:])
            new_scale = broadcast_to(tensor_scale, new_shape_scale, final_shape_scale, algorithms_scale, next_tensors_scale, self.manager)
            updated_algorithms.extend(algorithms_scale)
            new_shape_scale = final_shape_scale

        if len(shape_in) == 1:
            size_x_in = shape_in[0]
            size_y_in = 1
            size_z_in = 1
        elif len(shape_in) == 2:
            size_x_in = shape_in[0]
            size_y_in = shape_in[1]
            size_z_in = 1
        else:
            size_x_in = np.prod(shape_in[:-2])
            size_y_in = shape_in[-2]
            size_z_in = shape_in[-1]

        if len(new_shape_rms) == 1:
            size_x_rms = new_shape_rms[0]
            size_y_rms = 1
            size_z_rms = 1
        elif len(new_shape_rms) == 2:
            size_x_rms = new_shape_rms[0]
            size_y_rms = new_shape_rms[1]
            size_z_rms = 1
        else:
            size_x_rms = np.prod(new_shape_rms[:-2])
            size_y_rms = new_shape_rms[-2]
            size_z_rms = new_shape_rms[-1]

        if len(new_shape_scale) == 1:
            size_x_scale = new_shape_scale[0]
            size_y_scale = 1
            size_z_scale = 1
        elif len(new_shape_scale) == 2:
            size_x_scale = new_shape_scale[0]
            size_y_scale = new_shape_scale[1]
            size_z_scale = 1
        else:
            size_x_scale = np.prod(new_shape_scale[:-2])
            size_y_scale = new_shape_scale[-2]
            size_z_scale = new_shape_scale[-1]

        size = np.prod(shape_in)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (size_x_in, size_y_in, size_z_in)

        updated_algorithms.append(self.manager.algorithm([tensor_in, new_rms, new_scale, tensor_out],
                                                         self.compiled_shader_matmul,
                                                         workgroup,
                                                         [size_x_in, size_y_in, size_z_in,
                                                          size_x_rms, size_y_rms, size_z_rms,
                                                          size_x_scale, size_y_scale, size_z_scale],
                                                         []))

        return [(tensor_out, shape_in)]