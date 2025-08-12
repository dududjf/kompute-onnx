import kp
import numpy as np
from .shader_utils import compile_source

class CastLikeOp:
    """
    ONNX::CastLike 的 Kompute 实现
    现支持：float32 -> float32 / int32，int32 -> float32（通过先转 float32 的方式）
    """

    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        props = manager.get_device_properties()
        max_invocations = props['max_work_group_invocations']
        max_wg_size = props['max_work_group_size']

        local_size_x = 1
        while 2 * local_size_x <= max_invocations and 2 * local_size_x <= max_wg_size[0]:
            local_size_x *= 2
        self.local_size_x = local_size_x

        self.total_elems = None
        self.workgroup = None
        self.shader = None
        self.target_dtype = None

        # 关键点：
        # 1) 显式 layout(std430)
        # 2) 使用 uint spec constants（N、dtype_id 均为 uint）
        self.shader_code = """
#version 450
layout (local_size_x = {local_size_x}) in;

layout (std430, set = 0, binding = 0) readonly buffer InBuf    {{ float in_buf[]; }};
layout (std430, set = 0, binding = 1) writeonly buffer OutBufF {{ float out_f[]; }};
layout (std430, set = 0, binding = 2) writeonly buffer OutBufI {{ int   out_i[]; }};

layout (push_constant) uniform PC {{
    float N_f;         // 元素总数
    float dtype_id_f;  // 0.0=f32, 1.0=i32
}} pc;

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    uint N = uint(pc.N_f + 0.5);
    uint dtype_id = uint(pc.dtype_id_f + 0.5);
    if (idx >= N) return;

    if (dtype_id == 0u) {{
        out_f[idx] = in_buf[idx];
    }} else {{
        out_i[idx] = int(in_buf[idx]);   // toward-zero
    }}
}}

"""
    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"CastLikeOp({dev})"

    __str__ = __repr__

    def run(self, tensor_in: np.ndarray, tensor_like: np.ndarray):
        assert isinstance(tensor_in, np.ndarray), "CastLikeOp expects numpy ndarray as first input"
        assert isinstance(tensor_like, np.ndarray), "CastLikeOp expects numpy ndarray as second input"

        # —— 输入统一转成 float32 再进入 shader ——
        #   （i32->f32 的场景也可以这样做：先在 CPU 转成 float32，shader 里只处理 float -> {float,int}）
        x_flat = tensor_in.reshape(-1)
        if x_flat.dtype != np.float32:
            x_flat = x_flat.astype(np.float32, copy=False)
        N = x_flat.size

        # 目标 dtype → shader 分支 id
        if tensor_like.dtype == np.float32:
            dtype_id = np.uint32(0)
            want_int_output = False
        elif tensor_like.dtype == np.int32:
            dtype_id = np.uint32(1)
            want_int_output = True
        else:
            raise NotImplementedError(f"CastLikeOp only supports float32/int32 as target, got {tensor_like.dtype}")

        # 编译与调度参数（当 N 或目标 dtype 改变时）
        if self.shader is None or self.total_elems != N or self.target_dtype != int(dtype_id):
            self.total_elems = N
            self.target_dtype = int(dtype_id)
            local_size_x = min(self.local_size_x, max(1, N))
            self.shader = compile_source(self.shader_code.format(local_size_x=local_size_x))
            self.workgroup = ((N + local_size_x - 1) // local_size_x, 1, 1)

        # 分配 GPU buffer
        tensor_in = self.manager.tensor(x_flat.astype(np.float32, copy=False))
        tensor_out_f = self.manager.tensor(np.zeros(N, dtype=np.float32))
        tensor_out_i = self.manager.tensor(np.zeros(N, dtype=np.int32))

        # 注意：spec consts 必须与 constant_id 类型匹配
        algo = self.manager.algorithm(
            [tensor_in, tensor_out_f, tensor_out_i],
            self.shader,
            self.workgroup,
            [],  # spec const 空
            [float(N), float(dtype_id)]  # push consts，顺序与 PC 一致
        )

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out_f, tensor_out_i])) \
           .eval()

        if want_int_output:
            raw_i = tensor_out_i.data()
            # Kompute 的 Python 绑定常把 SSBO 映射为 float32 视图
            # 这里用 view(np.int32) 把同一段内存按 int32 解释
            if isinstance(raw_i, np.ndarray):
                output_tensor = (raw_i.view(np.int32)).reshape(tensor_in.shape)
            else:
                # 极少数绑定返回 bytes-like，用 frombuffer
                output_tensor = np.frombuffer(raw_i, dtype=np.int32, count=N).reshape(tensor_in.shape)
        else:
            output_tensor = tensor_out_f.data().reshape(tensor_in.shape)

        del tensor_in, tensor_out_f, tensor_out_i
        return [output_tensor]
