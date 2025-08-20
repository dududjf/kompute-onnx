import kp
import numpy as np
from .shader_utils import compile_source

# Hard_Sigmoid 默认常量
DEFAULT_ALPHA = float(0.2)
DEFAULT_BETA  = float(0.5)


class HardSigmoidOp:
    """
    onnx::HardSigmoid 的 Kompute 实现（逐元素一元 | float32）
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

layout (constant_id = 0) const float alpha = 0;
layout (constant_id = 1) const float beta  = 0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float x = in_buf[idx];
    float y = alpha * x + beta;
    if (y < 0.0) y = 0.0;
    else if (y > 1.0) y = 1.0;
    out_buf[idx] = y;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"HardSigmoidOp({dev})"

    __str__ = __repr__

    def run(self, inputs):
        # 允许 run([x])、run([x, alpha])、run([x, alpha, beta])
        x = inputs[0]

        #  指定alpha、gamma值，或使用 Hard_Sigmoid 默认常量
        alpha = float(inputs[1]) if len(inputs) > 1 else DEFAULT_ALPHA
        beta = float(inputs[2]) if len(inputs) > 2 else DEFAULT_BETA

        x_flat = x.reshape(-1).astype(np.float32)

        tensor_in = self.manager.tensor(x_flat)
        tensor_out = self.manager.tensor(np.zeros_like(x_flat))

        workgroup = (x_flat.size, 1, 1)
        algo = self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            [alpha, beta],
            []
        )

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        output_tensor = tensor_out.data().reshape(x.shape)
        del tensor_in, tensor_out
        return [output_tensor]