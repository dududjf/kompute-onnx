import kp
import numpy as np
from .shader_utils import compile_source


class ReluOp:
    """
    onnx::Relu 的 Kompute 实现（逐元素一元 | float32）
    """

    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        self.shader = compile_source("""
#version 450
layout (local_size_x = 1) in;

layout (set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float val = in_buf[idx];
    out_buf[idx] = val > 0.0 ? val : 0.0;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"ReluOp({dev})"

    __str__ = __repr__

    def run(self, inputs):
        # --- 1) 取入参并展平为 float32 1D buffer ---
        x = inputs[0]

        x_flat = x.reshape(-1).astype(np.float32)

        # --- 2) 准备 Kompute tensors ---
        tensor_in = self.manager.tensor(x_flat)
        tensor_out = self.manager.tensor(np.zeros_like(x_flat))

        # --- 3) 构建 algorithm & 执行序列 ---
        workgroup = (x_flat.size, 1, 1)
        algo = self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            [],
            []
        )

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        # --- 4) 还原形状并返回 ---
        output_tensor = tensor_out.data().reshape(x.shape)

        del tensor_in, tensor_out
        return [output_tensor]