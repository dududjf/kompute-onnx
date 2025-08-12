import kp
import numpy as np
from .shader_utils import compile_source


class ReluOp:
    """
    onnx::Relu 的 Kompute 实现（逐元素：out = max(in, 0)）
    - 输入：任意形状张量（当前实现假设 float32）
    - 输出：与输入同形状
    - 风格：与 MatMulOp 一致（spec const、sequence 三段式）
    """

    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        # 设备属性 -> 选择 1D local_size_x（尽量翻倍）
        props = manager.get_device_properties()
        max_invocations = props['max_work_group_invocations']
        max_wg_size = props['max_work_group_size']
        local_size_x = 1
        while 2 * local_size_x <= max_invocations and 2 * local_size_x <= max_wg_size[0]:
            local_size_x *= 2
        self.local_size_x = local_size_x

        # 运行期缓存
        self.total_elems = None
        self.workgroup = None
        self.shader = None

        # GLSL（注意 {{ }} 转义；用 spec const 传元素总数）
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

    float x = in_buf[idx];
    // ReLU：max(x, 0.0)
    out_buf[idx] = x > 0.0 ? x : 0.0;
}}
"""

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"ReluOp({dev})"

    __str__ = __repr__

    def run(self, x: np.ndarray):
        assert isinstance(x, np.ndarray), "ReluOp expects a numpy ndarray"

        # 展平到 1D，当前实现 float32
        x_flat = x.reshape(-1).astype(np.float32)
        N = x_flat.size

        # 创建 Kompute 张量
        t_in = self.manager.tensor(x_flat)
        t_out = self.manager.tensor(np.zeros_like(x_flat))

        # 形状变化时编译 shader、计算网格
        if self.shader is None or self.total_elems != N:
            self.total_elems = N
            local_size_x = min(self.local_size_x, max(1, N))
            self.shader = compile_source(
                self.shader_code.format(local_size_x=local_size_x)
            )
            self.workgroup = ((N + local_size_x - 1) // local_size_x, 1, 1)

        # 构建 algorithm & 执行
        algo = self.manager.algorithm(
            [t_in, t_out],
            self.shader,
            self.workgroup,
            [float(N)],   # spec const
            []            # push const
        )
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([t_out])) \
           .eval()

        y = t_out.data().reshape(x.shape)

        # 回收句柄
        del t_in, t_out
        return [y]
