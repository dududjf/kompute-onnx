import numpy as np
import kp
from .shader_utils import compile_source


class SliceOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(binding = 0) readonly  buffer in_buf     { float in_tensor[];     };
layout(binding = 1) writeonly buffer out_buf    { float out_tensor[];    };

layout(constant_id = 0) const float elements_per_slice_f = 0;  // 每个切片的元素数量
layout(constant_id = 1) const float start_f = 0;          // 切片的起始位置
layout(constant_id = 2) const float size_f = 0;           // 切片的大小
layout(constant_id = 3) const float axis_size_f = 0;      // 切片轴的大小
layout(constant_id = 4) const float step_f = 0;           // 步长

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    uint elements_per_slice = uint(elements_per_slice_f);
    uint start = uint(start_f);
    uint size = uint(size_f);
    uint axis_size = uint(axis_size_f);
    uint step = uint(step_f);

    // 计算输入和输出的偏移量
    uint in_base_offset = gx * axis_size * elements_per_slice;
    uint out_offset = gx * size * elements_per_slice + gy;

    // 计算输入索引 - 考虑步长
    uint axis_index = start + (gy / elements_per_slice) * step;
    uint inner_index = gy % elements_per_slice;
    uint in_offset = in_base_offset + axis_index * elements_per_slice + inner_index;

    out_tensor[out_offset] = in_tensor[in_offset];
}
""")

    def __repr__(self):
        return f"SliceOp({self.manager.get_device_properties()['device_name']})"

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
        output_tensors_and_shapes = self.fuse(input_tensors, updated_algorithms, updated_tensors)

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))

        # 只同步非空张量
        non_empty_tensors = [tensor for tensor, _ in output_tensors_and_shapes if tensor is not None]
        if non_empty_tensors:
            seq.record(kp.OpTensorSyncLocal(non_empty_tensors))
        seq.eval()

        outputs = []
        for tensor, shape in output_tensors_and_shapes:
            if tensor is not None:
                output = tensor.data().reshape(shape)
                outputs.append(output)
            else:
                # 创建空张量
                empty_array = np.array([], dtype=np.float32).reshape(shape)
                outputs.append(empty_array)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        starts = input_tensors[1][0].data().astype(int)
        ends = input_tensors[2][0].data().astype(int)
        axes = input_tensors[3][0].data().astype(int) if len(input_tensors) > 3 else None
        steps = input_tensors[4][0].data().astype(int) if len(input_tensors) > 4 else None

        if axes is None:
            axes = list(range(len(starts)))
        else:
            axes = [a + len(shape_in) if a < 0 else a for a in axes]

        # 设置指定轴的start和end
        for i, axis in enumerate(axes):
            # 处理负索引
            starts[i] = starts[i] + shape_in[i] if starts[i] < 0 else starts[i]
            starts[i] = max(0, min(starts[i], shape_in[axis]))
            ends[i] = ends[i] + shape_in[i] if ends[i] < 0 else ends[i]
            ends[i] = max(0, min(ends[i], shape_in[axis]))

        # 设置步长
        if steps is None:
            steps = [1] * len(axes)

        # 处理多个轴的切片（每次处理一个轴）
        tensor_out = tensor_in
        current_shape = [i for i in shape_in]

        for i in range(len(axes)):
            axis = axes[i]

            # 获取当前轴的start, end和step
            start = starts[i]
            end = ends[i]
            step = steps[i]

            # 计算切片大小
            if step > 0:
                size = max(0, (end - start + step - 1) // step)
            else:
                size = max(0, (start - end + (-step) - 1) // (-step))

            # 创建输出形状
            shape_out = [i for i in current_shape]
            shape_out[axis] = size

            # 计算预元素和每个切片的元素
            pre_elements = np.prod(shape_out[:axis]) if axis > 0 else 1
            elements_per_slice = np.prod(shape_out[axis + 1:]) if axis + 1 < len(shape_out) else 1
            pre_elements = int(pre_elements)
            elements_per_slice = int(elements_per_slice)

            # 计算总元素数量
            total_elements = int(pre_elements * size * elements_per_slice)

            if total_elements == 0:
                # 对于大小为0的情况，创建空张量
                return [(None, shape_out)]

            tensor_in = tensor_out
            # 创建输出张量
            tensor_out = self.manager.tensor(np.zeros(total_elements, dtype=np.float32))
            updated_tensors.append(tensor_out)

            # 设置工作组大小
            workgroup = (pre_elements, size * elements_per_slice, 1)

            # 创建并添加算法
            updated_algorithms.append(
                self.manager.algorithm(
                    [tensor_in, tensor_out],
                    self.compiled_shader,
                    workgroup,
                    [elements_per_slice, start, size, current_shape[axis], step],
                    []
                )
            )

            # 更新当前张量和形状，用于处理多个轴
            current_shape = shape_out

        # 返回最终结果
        return [(tensor_out, current_shape)]
