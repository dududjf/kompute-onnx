import numpy as np
import kp
from .shader_utils import compile_source

DEFAULT_KEEPDIMS = float(1.0)
DEFAULT_NOOP_WITH_EMPTY_AXES = float(0.0)

class ReduceMeanOp:

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source(
            """
#version 450
layout (local_size_x = 1, local_size_y = 1) in;
layout (binding = 0) buffer buf_in_tensor { float in_tensor[]; };   // 输入张量
layout (binding = 1) buffer buf_out_tensor { float out_tensor[]; }; // 输出张量
layout (binding = 2) buffer buf_sum_tensor { float sum_tensor[]; }; // 中间求和张量
layout (constant_id = 0) const float block_size_f = 0;  // 归约维度的大小
layout (constant_id = 1) const float in_block_f = 0;    // 输入块大小
layout (constant_id = 2) const float out_block_f = 0;   // 输出块大小

void main()
{
    uint gx = gl_GlobalInvocationID.x;  // 块组索引
    uint gy = gl_GlobalInvocationID.y;  // 归约维度内的索引

    uint block_size = uint(block_size_f);
    uint in_block = uint(in_block_f);
    uint out_block = uint(out_block_f);
    
    // 计算输入和中间张量的偏移量
    uint in_offset = gx * in_block + gy;
    uint sum_offset = gx * out_block;

    // 累加归约维度上的元素（原子操作确保线程安全）
    //atomicAdd(sum_tensor[sum_offset], in_tensor[in_offset]);
    sum_tensor[sum_offset] += in_tensor[in_offset];

    // 最后一个线程负责计算平均值
    if (gy == block_size - 1) {
        out_tensor[sum_offset] = sum_tensor[sum_offset] / block_size;
    }
}
""")

    def __repr__(self):
        return f"ReduceMeanOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _normalize_axes(axes, rank):
        if axes is None:
            return list(range(rank))
        a = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]
        return sorted(((x + rank) % rank for x in a)) if len(a) else []

    def run(self, *inputs):
        assert len(inputs) >= 1, "ReduceMeanOp needs at least x"
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
        assert len(input_tensors) >= 1, "ReduceMeanOp needs at least x"

        tensor_data = input_tensors[0][0]
        shape_data = input_tensors[0][1]
        axes = input_tensors[1][0].data().astype(int) if len(input_tensors) > 1 else None
        keepdims = float(input_tensors[2][0].data()) if len(input_tensors) > 2 else DEFAULT_KEEPDIMS
        noop_with_empty_axes = float(input_tensors[3][0].data()) if len(input_tensors) > 3 else DEFAULT_NOOP_WITH_EMPTY_AXES

        if noop_with_empty_axes and axes is None:
            return [(tensor_data, shape_data)]

        # 规范化轴索引
        normalized_axes = self._normalize_axes(axes, len(shape_data))

        # 初始化变量，借鉴broadcast_to的分维度处理思想
        tensor_out = tensor_data
        current_shape = shape_data.copy()
        axes_set = set(normalized_axes)
        block_size = 1  # 基础块大小（非归约维度的乘积）
        end = len(current_shape) - 1  # 从最后一个维度开始处理

        # 从后往前遍历维度，类似broadcast_to的反向操作
        while end >= 0:
            start = end
            # 检查当前维度是否需要归约
            if end in axes_set:
                # 向前寻找连续的归约维度（合并处理提高效率）
                while start >= 0 and start in axes_set:
                    start -= 1

                # 计算归约所需的参数
                # 非归约部分的块组数（前start+1维的乘积）
                group_x = np.prod(current_shape[:start + 1]) if start >= 0 else 1
                # 归约维度的总大小
                reduce_size = np.prod(current_shape[start + 1:end + 1])
                # 输入块大小：归约维度大小 × 基础块大小
                in_block = reduce_size * block_size
                # 输出块大小：基础块大小（归约后维度被移除或保留为1）
                out_block = block_size

                # 创建中间张量（用于累加求和）和输出张量
                sum_array = np.zeros(group_x * out_block, dtype=np.float32)
                sum_tensor = self.manager.tensor(sum_array)
                out_array = np.zeros(group_x * out_block, dtype=np.float32)
                tensor_out = self.manager.tensor(out_array)

                # 配置GPU算法
                workgroup = (group_x, reduce_size, 1)
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_data, tensor_out, sum_tensor],
                    self.shader,
                    workgroup,
                    [reduce_size, in_block, out_block],
                    []
                ))

                # 更新参数，准备处理下一个维度
                updated_tensors.extend([sum_tensor, tensor_out])
                tensor_in = tensor_out

                # 更新基础块大小
                block_size = out_block

                # 根据keepdim决定是否保留归约后的维度
                if keepdims:
                    # 保留维度，在形状中插入1
                    reduced_dims = [1] * (end - start)
                    current_shape = current_shape[:start + 1] + reduced_dims + current_shape[end + 1:]
                else:
                    # 移除归约维度
                    current_shape = current_shape[:start + 1] + current_shape[end + 1:]

                end = start  # 继续处理前面的维度
            else:
                # 非归约维度，累加到基础块大小
                block_size *= current_shape[end]
                end -= 1

        return [(tensor_out, current_shape)]
