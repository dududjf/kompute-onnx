import kp
import numpy as np
from .shader_utils import compile_source

# ONNX Softmax 默认 axis=1
DEFAULT_AXIS = 1


class SoftmaxOp:
    """
    onnx::Softmax 的 Kompute 实现
    """

    def __init__(self, manager: kp.Manager):
        self.manager = manager

        # 1) 沿 axis reduce 最大值: 输出 buffer 大小 = STRIDE_BEFORE * STRIDE_AFTER
        self.reduce_max_shader = compile_source(r"""
#version 450
layout(local_size_x = 1) in;
layout(binding = 0) readonly buffer InBuf { float in_buf[]; };
layout(binding = 1) writeonly buffer OutMax { float out_max[]; };
layout(constant_id = 0) const float AXIS_SIZE_F     = 0.0;
layout(constant_id = 1) const float STRIDE_BEFORE_F = 0.0;
layout(constant_id = 2) const float STRIDE_AFTER_F  = 0.0;

void main(){
    uint out_idx       = gl_GlobalInvocationID.x;        // 0 .. STRIDE_BEFORE*STRIDE_AFTER-1
    uint AXIS_SIZE     = uint(AXIS_SIZE_F);
    uint STRIDE_BEFORE = uint(STRIDE_BEFORE_F);
    uint STRIDE_AFTER  = uint(STRIDE_AFTER_F);

    // out_idx <-> (before, after)
    uint before = out_idx / STRIDE_AFTER;
    uint after  = out_idx % STRIDE_AFTER;

    // 输入中该组的起点（axis 上第0个）
    uint base = before * AXIS_SIZE * STRIDE_AFTER + after;

    float m = in_buf[base];
    for(uint i=1u;i<AXIS_SIZE;++i){
        float v = in_buf[base + i*STRIDE_AFTER];
        if (v > m) m = v;
    }
    out_max[out_idx] = m;
}
""")

        # 2) 逐元素: y = exp(x - max[group])
        self.sub_exp_shader = compile_source(r"""
#version 450
layout(local_size_x = 1) in;
layout(binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout(binding = 1) readonly buffer MaxBuf { float max_buf[]; };
layout(binding = 2) writeonly buffer OutY  { float y_buf[]; };
layout(constant_id = 0) const float AXIS_SIZE_F     = 0.0;
layout(constant_id = 1) const float STRIDE_BEFORE_F = 0.0;
layout(constant_id = 2) const float STRIDE_AFTER_F  = 0.0;

void main(){
    uint idx           = gl_GlobalInvocationID.x;        // 0 .. numel-1
    uint AXIS_SIZE     = uint(AXIS_SIZE_F);
    uint STRIDE_BEFORE = uint(STRIDE_BEFORE_F);
    uint STRIDE_AFTER  = uint(STRIDE_AFTER_F);

    // idx -> (before, axis_i, after)
    uint group   = idx / STRIDE_AFTER;     // = before*AXIS_SIZE + axis_i
    uint after   = idx % STRIDE_AFTER;
    uint before  = group / AXIS_SIZE;

    float mx = max_buf[before * STRIDE_AFTER + after];
    float x  = in_buf[idx];
    y_buf[idx] = exp(x - mx);
}
""")

        # 3) 沿 axis reduce 求和: 输出大小 = STRIDE_BEFORE * STRIDE_AFTER
        self.reduce_sum_shader = compile_source(r"""
#version 450
layout(local_size_x = 1) in;
layout(binding = 0) readonly buffer InY     { float y_buf[]; };
layout(binding = 1) writeonly buffer OutSum { float sum_buf[]; };
layout(constant_id = 0) const float AXIS_SIZE_F     = 0.0;
layout(constant_id = 1) const float STRIDE_BEFORE_F = 0.0;
layout(constant_id = 2) const float STRIDE_AFTER_F  = 0.0;

void main(){
    uint out_idx       = gl_GlobalInvocationID.x;
    uint AXIS_SIZE     = uint(AXIS_SIZE_F);
    uint STRIDE_BEFORE = uint(STRIDE_BEFORE_F);
    uint STRIDE_AFTER  = uint(STRIDE_AFTER_F);

    uint before = out_idx / STRIDE_AFTER;
    uint after  = out_idx % STRIDE_AFTER;
    uint base   = before * AXIS_SIZE * STRIDE_AFTER + after;

    float s = 0.0;
    for(uint i=0u;i<AXIS_SIZE;++i){
        s += y_buf[base + i*STRIDE_AFTER];
    }
    sum_buf[out_idx] = s;
}
""")

        # 4) 逐元素: out = y / sum[group]
        self.normalize_shader = compile_source(r"""
#version 450
layout(local_size_x = 1) in;
layout(binding = 0) readonly buffer InY     { float y_buf[]; };
layout(binding = 1) readonly buffer SumBuf  { float sum_buf[]; };
layout(binding = 2) writeonly buffer OutBuf { float out_buf[]; };
layout(constant_id = 0) const float AXIS_SIZE_F     = 0.0;
layout(constant_id = 1) const float STRIDE_BEFORE_F = 0.0;
layout(constant_id = 2) const float STRIDE_AFTER_F  = 0.0;

void main(){
    uint idx           = gl_GlobalInvocationID.x;
    uint AXIS_SIZE     = uint(AXIS_SIZE_F);
    uint STRIDE_BEFORE = uint(STRIDE_BEFORE_F);
    uint STRIDE_AFTER  = uint(STRIDE_AFTER_F);

    uint group   = idx / STRIDE_AFTER;     // = before*AXIS_SIZE + axis_i
    uint after   = idx % STRIDE_AFTER;
    uint before  = group / AXIS_SIZE;

    float denom = sum_buf[before * STRIDE_AFTER + after];
    float y     = y_buf[idx];
    out_buf[idx] = (denom == 0.0) ? 0.0 : (y / denom);
}
""")

    def __repr__(self):
        return f"SoftmaxOp({self.manager.get_device_properties()['device_name']})"

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
        x_tensor, x_shape = input_tensors[0]
        axis = int(input_tensors[1][0].data()) if len(input_tensors) > 1 and input_tensors[1][0] is not None else DEFAULT_AXIS
        rank = len(x_shape)
        if axis < 0:
            axis += rank
        assert 0 <= axis < rank, "axis out of range"

        axis_size     = int(x_shape[axis])
        stride_before = int(np.prod(x_shape[:axis])) if axis > 0 else 1
        stride_after  = int(np.prod(x_shape[axis+1:])) if axis < rank-1 else 1
        numel         = int(np.prod(x_shape)) if len(x_shape) else 1
        groups        = int(stride_before * stride_after)  # reduce 输出大小

        # max_buf: groups
        max_buf = self.manager.tensor(np.zeros(groups, dtype=np.float32))
        updated_tensors.append(max_buf)
        # y_buf: numel
        y_buf   = self.manager.tensor(np.zeros(numel, dtype=np.float32))
        updated_tensors.append(y_buf)
        # sum_buf: groups
        sum_buf = self.manager.tensor(np.zeros(groups, dtype=np.float32))
        updated_tensors.append(sum_buf)
        # out: numel
        out_buf = self.manager.tensor(np.zeros(numel, dtype=np.float32))
        updated_tensors.append(out_buf)

        spec = [float(axis_size), float(stride_before), float(stride_after)]

        # 1) reduce_max
        updated_algorithms.append(
            self.manager.algorithm([x_tensor, max_buf],
                                   self.reduce_max_shader,
                                   (groups,1,1), spec, [])
        )
        # 2) sub_exp
        updated_algorithms.append(
            self.manager.algorithm([x_tensor, max_buf, y_buf],
                                   self.sub_exp_shader,
                                   (numel,1,1), spec, [])
        )
        # 3) reduce_sum
        updated_algorithms.append(
            self.manager.algorithm([y_buf, sum_buf],
                                   self.reduce_sum_shader,
                                   (groups,1,1), spec, [])
        )
        # 4) normalize
        updated_algorithms.append(
            self.manager.algorithm([y_buf, sum_buf, out_buf],
                                   self.normalize_shader,
                                   (numel,1,1), spec, [])
        )

        return [(out_buf, x_shape)]