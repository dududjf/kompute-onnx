import kp
import numpy as np
from .shader_utils import compile_source

DEFAULT_AXIS = 0
DEFAULT_KEEPDIMS = True # 是否保持维度
DEFAULT_SELECT_LAST_INDEX = False  # False=第一个最小值, True=最后一个最小值

class ArgMinOp:
    """
    onnx::ArgMin 的 Kompute 实现（沿指定轴找到最小值的索引）
    """

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout(local_size_x=1, local_size_y=1) in;

layout(binding=0) readonly buffer InBuf  { float in_buf[]; };
layout(binding=1) writeonly buffer OutBuf{ int   out_buf[]; };

layout(constant_id=0) const float axis_size_f    = 0.0;
layout(constant_id=1) const float stride_after_f = 0.0;
layout(constant_id=2) const float select_last_f  = 0.0;

void main() {
    uint axis_size    = uint(axis_size_f);
    uint stride_after = uint(stride_after_f);
    uint select_last  = uint(select_last_f);

    uint out_before = gl_GlobalInvocationID.x;
    uint out_after  = gl_GlobalInvocationID.y;

    uint base = (out_before * axis_size) * stride_after + out_after;

    float min_val = in_buf[base];
    uint  min_idx = 0u;
    base += stride_after;

    if (select_last == 1u) {
        for (uint i = 1u; i < axis_size; ++i, base += stride_after) {
            float v = in_buf[base];
            if (v <= min_val) {
                min_val = v;
                min_idx = i;
            }
        }
    } 
    else {
        for (uint i = 1u; i < axis_size; ++i, base += stride_after) {
            float v = in_buf[base];
            if (v < min_val) {
                min_val = v;
                min_idx = i;
            }
        }
    }
    out_buf[out_before * stride_after + out_after] = int(min_idx);
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"ArgMinOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
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
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        print(tensor_out.data().dtype)
        output = tensor_out.data().reshape(output_shape).astype(np.int64)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape = input_tensors[0]
        rank = len(shape)
        axis = int(input_tensors[1][0].data().reshape(-1)[0]) if len(input_tensors) > 1 \
            else DEFAULT_AXIS
        keepdims = int(input_tensors[2][0].data().reshape(-1)[0]) if len(input_tensors) > 2 \
            else DEFAULT_KEEPDIMS
        select_last_index = int(input_tensors[3][0].data().reshape(-1)[0]) if len(input_tensors) > 3 \
            else DEFAULT_SELECT_LAST_INDEX

        if axis < 0:
            axis = rank + axis
        assert 0 <= axis < rank, "axis out of range"

        axis_size = int(shape[axis])
        stride_before = int(np.prod(shape[:axis])) if axis > 0 else 1
        stride_after  = int(np.prod(shape[axis+1:])) if axis < rank - 1 else 1

        out_shape = list(shape)
        out_shape[axis:axis + 1] = [1] if keepdims else []

        output_size = stride_before * stride_after
        if output_size <= 0:
            output_size = 1

        tensor_out = self.manager.tensor_t(np.zeros(output_size, dtype=np.int32), tensor_type=kp.TensorTypes.device)
        updated_tensors.append(tensor_out)
        workgroup = (stride_before, stride_after, 1)
        shader = self.shader
        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_out],
                shader,
                workgroup,
                [axis_size, stride_after, select_last_index],
                []
            )
        )

        return [(tensor_out, out_shape)]
