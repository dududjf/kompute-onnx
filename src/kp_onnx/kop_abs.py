import kp
import numpy as np
from .shader_utils import compile_source


class AbsOp:
    """
    onnx::Abs 的 Kompute 实现（逐元素一元 | float32）
    复用：设备属性获取、local_size 自动选择、GLSL->SPIR-V、algorithm/sequence 管线、形状驱动的重编译
    专用：shader 核心逻辑 out[i] = abs(in[i])；1D 网格；spec const 传元素总数 N
    """

    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        # [通用] 设备属性与 local_size 自适应（1D 网格）
        props = manager.get_device_properties()
        max_invocations = props['max_work_group_invocations']
        max_wg_size = props['max_work_group_size']

        local_size_x = 1
        while 2 * local_size_x <= max_invocations and 2 * local_size_x <= max_wg_size[0]:
            local_size_x *= 2
        self.local_size_x = local_size_x

        # [通用缓存]
        self.total_elems = None
        self.workgroup = None
        self.shader = None

        # [Abs 专用] 逐元素 abs 的 GLSL（注意 {{ }} 转义花括号）
        self.shader_code = """
#version 450
layout (local_size_x = {local_size_x}) in;

layout (std430, set = 0, binding = 0) readonly buffer InBuf  {{ float in_buf[]; }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf {{ float out_buf[]; }};

layout (constant_id = 0) const uint N = 0u;  // 元素总数（用 uint 避免精度丢失）

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= N) return;

    // [Abs 专用] 核心：逐元素绝对值
    out_buf[idx] = abs(in_buf[idx]);
}}
"""

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"AbsOp({dev})"

    __str__ = __repr__

    def run(self, x: np.ndarray):
        # [通用] 入参断言与扁平化；当前实现用 float32
        assert isinstance(x, np.ndarray), "AbsOp expects a numpy ndarray"
        x_flat = x.reshape(-1).astype(np.float32)
        N = x_flat.size

        # [通用] 创建当次 tensors（如果追求性能，可改为长生命周期复用）
        t_in = self.manager.tensor(x_flat)
        t_out = self.manager.tensor(np.zeros_like(x_flat))

        # [通用] 形状变化触发重编译/重建调度
        if self.shader is None or self.total_elems != N:
            self.total_elems = N
            local_size_x = min(self.local_size_x, max(1, N))
            self.shader = compile_source(self.shader_code.format(local_size_x=local_size_x))
            self.workgroup = ((N + local_size_x - 1) // local_size_x, 1, 1)

        # [通用] 构建 algorithm & 执行序列
        algo = self.manager.algorithm(
            [t_in, t_out],
            self.shader,
            self.workgroup,
            [np.uint32(N)],   # 用 spec const 传 N（uint32）
            []
        )
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([t_out])) \
           .eval()

        y = t_out.data().reshape(x.shape)
        del t_in, t_out
        return [y]
