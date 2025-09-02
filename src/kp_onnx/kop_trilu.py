import kp
import numpy as np
from .shader_utils import compile_source

DEFAULT_UPPER = 1  # onnx: 默认取上三角
DEFAULT_K = 0      # onnx: 默认 k=0

class TriluOp:
    """
    onnx::Trilu 的 Kompute 实现（最后两维按三角掩码，其余批次维逐批处理 | float32）
    """

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout (local_size_x = 1) in;

layout (set = 0, binding = 0) readonly  buffer InBuf  { float in_buf[];  };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout (constant_id = 0) const float N_f = 0.0;   // 最后两维中的 N(行)
layout (constant_id = 1) const float M_f = 0.0;   // 最后两维中的 M(列)
layout (constant_id = 2) const float K_f = 0.0;   // 偏移 k（可正可负）
layout (constant_id = 3) const float U_f = 1.0;   // upper 标志: 1=upper, 0=lower

void main() {
    uint idx = gl_GlobalInvocationID.x;

    uint N = uint(N_f);
    uint M = uint(M_f);
    int  k = int(K_f);
    bool upper = (U_f > 0.5);

    // 还原到最后两维坐标 (i, j)
    uint block = N * M;             // 单个矩阵的元素数
    uint rem   = idx % block;       // 在该矩阵内的线性偏移
    uint i     = rem / M;           // 行
    uint j     = rem % M;           // 列

    // 判定是否保留
    bool keep = upper ? (int(j) - int(i) >= k) : (int(i) - int(j) >= -k);

    float x = in_buf[idx];
    out_buf[idx] = keep ? x : 0.0;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"TriluOp({dev})"
    __str__ = __repr__

    def run(self, *inputs):
        # inputs: input[, k_scalar][, upper_scalar]
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
        in_tensor, in_shape = input_tensors[0]
        assert len(in_shape) >= 2, "Trilu expects rank >= 2"
        N = in_shape[-2]
        M = in_shape[-1]
        k = input_tensors[1][0].data() if len(input_tensors) >= 2 else DEFAULT_K
        upper = input_tensors[2][0].data() if len(input_tensors) >= 3 else DEFAULT_UPPER
        size = np.prod(in_shape)

        out_tensor = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(out_tensor)

        spec_consts = [float(N), float(M), float(k), float(upper)]
        workgroup = (size, 1, 1)
        updated_algorithms.append(
            self.manager.algorithm(
                [in_tensor, out_tensor],
                self.shader,
                workgroup,
                spec_consts,
                []
            )
        )
        return [(out_tensor, in_shape)]