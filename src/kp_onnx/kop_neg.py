import kp
import numpy as np
from .shader_utils import compile_source


class NegOp:
    """
    onnx::Neg 的 Kompute 实现（逐元素一元取负，优化版）
    - Shader 只编译一次
    - Algorithm 只构建一次
    - GPU 缓冲区可复用
    """

    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        # [可复用] 设备属性与 local_size 自适应
        props = manager.get_device_properties()
        max_invocations = props['max_work_group_invocations']
        max_wg_size = props['max_work_group_size']

        local_size_x = 1
        while 2 * local_size_x <= max_invocations and 2 * local_size_x <= max_wg_size[0]:
            local_size_x *= 2
        self.local_size_x = local_size_x

        # 缓存
        self.total_elems = None
        self.workgroup = None
        self.shader = None
        self.t_in = None
        self.t_out = None
        self.algo = None

        # [Neg 专用] GLSL 核心逻辑
        self.shader_code = """
#version 450
layout (local_size_x = {local_size_x}) in;

layout (set = 0, binding = 0) readonly buffer InBuf  {{ float in_buf[]; }};
layout (set = 0, binding = 1) writeonly buffer OutBuf {{ float out_buf[]; }};

layout (constant_id = 0) const float N_f = 0;  // 元素总数

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    uint N = uint(N_f);
    if (idx >= N) return;
    out_buf[idx] = -in_buf[idx];
}}
"""

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"NegOp({dev})"

    __str__ = __repr__

    def _init_resources(self, N):
        """只在第一次或数据规模变化时初始化 GPU 资源"""
        self.total_elems = N
        local_size_x = min(self.local_size_x, max(1, N))

        # 编译 shader
        self.shader = compile_source(self.shader_code.format(local_size_x=local_size_x))

        # 计算 1D 网格尺寸
        self.workgroup = ((N + local_size_x - 1) // local_size_x, 1, 1)

        # 分配 GPU 缓冲区（一次性）
        self.t_in = self.manager.tensor(np.zeros(N, dtype=np.float32))
        self.t_out = self.manager.tensor(np.zeros(N, dtype=np.float32))

        # 构建 Algorithm（一次性）
        self.algo = self.manager.algorithm(
            [self.t_in, self.t_out],
            self.shader,
            self.workgroup,
            [float(N)],
            []
        )

    def run(self, x: np.ndarray):
        assert isinstance(x, np.ndarray), "NegOp expects a numpy ndarray"
        x_flat = x.reshape(-1).astype(np.float32)
        N = x_flat.size

        if self.shader is None or self.total_elems != N:
            self._init_resources(N)

        # 将数据写入已分配的 GPU 缓冲区
        self.t_in.data()[:] = x_flat

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([self.t_in])) \
           .record(kp.OpAlgoDispatch(self.algo)) \
           .record(kp.OpTensorSyncLocal([self.t_out])) \
           .eval()

        return [self.t_out.data().reshape(x.shape)]
