import kp
import numpy as np
from .shader_utils import compile_source


class ExpOp:
    """
    onnx::Exp 的 Kompute 实现（逐元素一元指数）
    - 输入：任意形状张量（当前实现假设 float32）
    - 输出：与输入同形状
    复用点：
      * 设备属性查询 / local_size 自动选择
      * GLSL -> SPIR-V 编译、algorithm 构建
      * sequence: SyncDevice -> Dispatch -> SyncLocal
      * 形状变化触发重编译（spec const / workgroup 缓存）
    专用点：
      * shader 核心逻辑：out[i] = exp(in[i])
      * spec const 仅需元素总数（N）
      * 形状校验/推断：输出形状=输入形状
    """

    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        # —— 可复用：选择合适的 local_size_x（1D 网格足够）——
        props = manager.get_device_properties()
        max_invocations = props['max_work_group_invocations']
        max_wg_size = props['max_work_group_size']

        local_size_x = 1
        while 2 * local_size_x <= max_invocations and 2 * local_size_x <= max_wg_size[0]:
            local_size_x *= 2
        self.local_size_x = local_size_x

        # —— 运行期缓存 ——（形状变才重建）
        self.total_elems = None
        self.workgroup = None
        self.shader = None

        # —— 专用：GLSL compute shader（逐元素 exp）——
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

    // —— 核心：逐元素指数 ——
    out_buf[idx] = exp(in_buf[idx]);
}}
"""

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"ExpOp({dev})"

    __str__ = __repr__

    def run(self, x: np.ndarray):
        # —— 可复用：基础断言+展平到 1D ——
        assert isinstance(x, np.ndarray), "ExpOp expects a numpy ndarray"
        x_flat = x.reshape(-1).astype(np.float32)
        N = x_flat.size

        # —— 可复用：创建 Kompute 张量 ——
        t_in = self.manager.tensor(x_flat)
        t_out = self.manager.tensor(np.zeros_like(x_flat))

        # —— 可复用：形状变化才重新编译/建图 ——
        if self.shader is None or self.total_elems != N:
            self.total_elems = N
            local_size_x = min(self.local_size_x, max(1, N))

            # 编译 shader
            self.shader = compile_source(self.shader_code.format(local_size_x=local_size_x))
            # 计算 1D 网格
            self.workgroup = ((N + local_size_x - 1) // local_size_x, 1, 1)

        # —— 可复用：执行（设备同步→调度→取回） ——
        algo = self.manager.algorithm(
            [t_in, t_out],
            self.shader,
            self.workgroup,
            [float(N)],  # spec const：元素数
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
