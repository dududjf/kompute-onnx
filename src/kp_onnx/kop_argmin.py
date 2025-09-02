import kp
import numpy as np
from .shader_utils import compile_source

DEFAULT_AXIS = 0
DEFAULT_KEEPDIMS = 1

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
layout(binding=1) writeonly buffer OutBuf{ float out_buf[]; };

layout(constant_id=0) const float AXIS_SIZE_F     = 0.0;
layout(constant_id=1) const float STRIDE_BEFORE_F = 0.0;
layout(constant_id=2) const float STRIDE_AFTER_F  = 0.0;

void main() {
    uint AXIS_SIZE     = uint(AXIS_SIZE_F);
    uint STRIDE_BEFORE = uint(STRIDE_BEFORE_F);
    uint STRIDE_AFTER  = uint(STRIDE_AFTER_F);

    uint out_before = gl_GlobalInvocationID.x;
    uint out_after  = gl_GlobalInvocationID.y;
    if (out_before >= STRIDE_BEFORE || out_after >= STRIDE_AFTER) return;

    uint base = out_before * AXIS_SIZE * STRIDE_AFTER + out_after;

    float min_val = in_buf[base];
    uint  min_idx = 0u;
    for (uint i = 1u; i < AXIS_SIZE; ++i) {
        float v = in_buf[base + i * STRIDE_AFTER];
        if (v < min_val) { min_val = v; min_idx = i; }
    }
    out_buf[out_before * STRIDE_AFTER + out_after] = float(min_idx);
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"ArgMinOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            if inp is None:
                tensor = None
            else:
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

        output = tensor_out.data().reshape(output_shape).astype(np.int64)

        for tensor, _ in input_tensors:
            if tensor is not None:
                del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape = input_tensors[0]
        rank = len(shape)
        axis = int(input_tensors[1][0].data().reshape(-1)[0]) if input_tensors[1][0] is not None else DEFAULT_AXIS
        keepdims = int(input_tensors[2][0].data().reshape(-1)[0]) if input_tensors[2][0] is not None else DEFAULT_KEEPDIMS

        if axis < 0:
            axis = rank + axis
        assert 0 <= axis < rank, "axis out of range"

        axis_size = int(shape[axis])
        assert axis_size > 0, "ArgMin requires the reduction axis to have size > 0"

        stride_before = int(np.prod(shape[:axis])) if axis > 0 else 1
        stride_after  = int(np.prod(shape[axis+1:])) if axis < rank - 1 else 1

        # ---- 输出形状 ----
        if keepdims:
            out_shape = list(shape)
            out_shape[axis] = 1
        else:
            out_shape = [d for i, d in enumerate(shape) if i != axis]

        output_size = stride_before * stride_after
        if output_size <= 0:
            output_size = 1

        tensor_out = self.manager.tensor(np.zeros(output_size, dtype=np.float32))
        updated_tensors.append(tensor_out)
        workgroup = (stride_before, stride_after, 1)
        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_out],
                self.shader,
                workgroup,
                [float(axis_size), float(stride_before), float(stride_after)],  # ← 用 float
                []
            )
        )

        return [(tensor_out, out_shape)]
