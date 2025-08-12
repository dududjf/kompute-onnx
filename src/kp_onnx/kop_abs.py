import kp
import numpy as np
from .shader_utils import compile_source


class AbsOp:
    """
    onnx::Abs 的 Kompute 实现（逐元素一元 | float32）
    """

    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        # [可复用] 根据设备属性选择合适的 local_size_x
        props = manager.get_device_properties()
        max_invocations = props['max_work_group_invocations']
        max_wg_size = props['max_work_group_size']

        local_size_x = 1
        # 一元逐元素 → 1D 网格
        while 2 * local_size_x <= max_invocations and 2 * local_size_x <= max_wg_size[0]:
            local_size_x *= 2
        self.local_size_x = local_size_x

        # [可复用] 运行期缓存
        self.total_elems = None
        self.workgroup = None
        self.shader = None

        # [Abs 专用] 逐元素 abs 的 GLSL
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

        # [可复用] 展平到 1D buffer，当前实现 float32
        x_flat = x.reshape(-1).astype(np.float32)
        N = x_flat.size

        # [可复用] 构建/复用 GPU tensors
        tensor_in = self.manager.tensor(x_flat)
        tensor_out = self.manager.tensor(np.zeros_like(x_flat))

        # [可复用] 形状变化触发重编译与重建调度参数
        if self.shader is None or self.total_elems != N:
            self.total_elems = N
            # 选择 workgroup 大小（1D）
            local_size_x = min(self.local_size_x, max(1, N))
            # 编译 shader
            self.shader = compile_source(self.shader_code.format(local_size_x=local_size_x))
            # 计算 1D 网格尺寸
            self.workgroup = ((N + local_size_x - 1) // local_size_x, 1, 1)

        # [通用] 构建 algorithm & 执行序列
        algo = self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            self.workgroup,
            [int(N)],   # 用 spec const 传 N（f32）
            []
        )
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        # [可复用] 还原形状并返回
        output_tensor = tensor_out.data().reshape(x.shape)

        # [可复用] 资源句柄释放（有助于更及时地回收底层资源）
        del tensor_in, tensor_out
        return [output_tensor]
