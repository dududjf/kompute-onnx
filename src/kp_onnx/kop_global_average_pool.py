import kp
import numpy as np
from .shader_utils import compile_source


class GlobalAveragePoolOp:
    """
    onnx::GlobalAveragePool 的 Kompute 实现（N,C,spatial... → N,C,1,...,1 | float32）
    """

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source(r"""
#version 450
layout (local_size_x = 1) in;

layout (set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout (constant_id = 0) const float Nf = 0;
layout (constant_id = 1) const float Cf = 0;
layout (constant_id = 2) const float Sf = 0;             // S = ∏(空间维)
layout (constant_id = 3) const float STRIDE_N_f = 0;     // = C * S
layout (constant_id = 4) const float STRIDE_C_f = 0;     // = S

void main() {
    uint N = uint(Nf);
    uint C = uint(Cf);
    uint S = uint(Sf);
    uint STRIDE_N = uint(STRIDE_N_f);
    uint STRIDE_C = uint(STRIDE_C_f);

    uint gid = gl_GlobalInvocationID.x;
    if (gid >= N * C) return;

    // 映射到 (n,c)
    uint n = gid / C;
    uint c = gid % C;

    // 该 (n,c) 在扁平内存中的起始位置
    uint base = n * STRIDE_N + c * STRIDE_C;

    // 空间维在线性内存上连续，直接累加 S 个元素
    float acc = 0.0;
    for (uint t = 0u; t < S; ++t) {
        acc += in_buf[base + t];
    }

    out_buf[gid] = (S == 0u) ? 0.0 : acc / float(S);
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"GlobalAveragePoolOp({dev})"

    __str__ = __repr__

    def run(self, inputs):
        x = inputs[0]
        # x = x.astype(np.float32)
        shape = list(x.shape)
        assert len(shape) >= 2, "GlobalAveragePool expects at least [N, C, ...]"

        N = int(shape[0])
        C = int(shape[1])
        S = int(np.prod(shape[2:])) if len(shape) > 2 else 1
        strideN = np.prod(shape[1:])  # C * S
        strideC = S

        # 输出形状：空间维全部变为 1
        out_shape = [N, C] + [1] * max(0, len(shape) - 2)
        total_out = int(np.prod(out_shape)) if len(out_shape) > 0 else 1  # = N*C

        # 空输出早退
        if total_out == 0:
            return [np.zeros(out_shape, dtype=np.float32)]

        # IO tensors
        x_flat = x.reshape(-1).astype(np.float32)
        tensor_in = self.manager.tensor(x_flat)
        tensor_out = self.manager.tensor(np.zeros(total_out, dtype=np.float32))

        # spec constants
        spec_consts = [float(N), float(C), float(S), float(strideN), float(strideC)]

        # 每个 (n,c) 一个线程
        workgroup = (int(N * C), 1, 1)
        algo = self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            spec_consts,
            []
        )

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        out = tensor_out.data().reshape(out_shape)
        del tensor_in, tensor_out
        return [out]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]],
             updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert len(input_tensors) == 2, "GlobalAveragePool expects at least [N, C, ...]"
        tensor_in, in_shape = input_tensors[0]

        N = int(in_shape[0])
        C = int(in_shape[1])
        S = int(np.prod(in_shape[2:])) if len(in_shape) > 2 else 1
        strideN = np.prod(in_shape[1:])  # C * S
        strideC = S

        out_shape = [N, C] + [1] * max(0, len(in_shape) - 2)
        total_out = int(np.prod(out_shape)) if len(out_shape) > 0 else 1

        tensor_out = self.manager.tensor(np.zeros(total_out, dtype=np.float32))
        updated_tensors.append(tensor_out)

        if total_out == 0:
            return [(tensor_out, out_shape)]

        spec_consts = [float(N), float(C), float(S), float(strideN), float(strideC)]

        algo = self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            (int(N * C), 1, 1),
            spec_consts,
            []
        )
        updated_algorithms.append(algo)
        return [(tensor_out, out_shape)]
