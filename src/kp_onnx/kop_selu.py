import kp
import numpy as np
from .shader_utils import compile_source


class SeluOp:
    """
    onnx::Selu 的 Kompute 实现（逐元素一元 | float32）
    """

    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        self.shader = compile_source("""
#version 450
layout (local_size_x = 1) in;

layout (std430, set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (std430, set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };
layout (std430, set = 0, binding = 2) readonly buffer Scalars { float scalars[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float alpha = scalars[0];
    float gamma = scalars[1];
    float x = in_buf[idx];
    // SELU(x) = gamma * ( x (x>=0) ; alpha*(exp(x)-1) (x<0) )
    float y = (x >= 0.0) ? x : (alpha * (exp(x) - 1.0));
    out_buf[idx] = gamma * y;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"SeluOp({dev})"

    __str__ = __repr__

    def run(self, inputs):
        # 允许 run([x])、run([x, alpha])、run([x, alpha, gamma])
        x = inputs[0]

        #  指定alpha、gamma值，或使用 SELU 默认常量（float32）
        alpha = np.float32(inputs[1]) if len(inputs) > 1 else np.float32(1.6732631921768188)
        gamma = np.float32(inputs[2]) if len(inputs) > 2 else np.float32(1.0507009873554805)

        x_flat = x.reshape(-1).astype(np.float32)

        tensor_in = self.manager.tensor(x_flat)
        tensor_out = self.manager.tensor(np.zeros_like(x_flat))
        tensor_scalars = self.manager.tensor(np.array([alpha, gamma], dtype=np.float32))

        workgroup = (x_flat.size, 1, 1)
        algo = self.manager.algorithm(
            [tensor_in, tensor_out, tensor_scalars],
            self.shader,
            workgroup,
            [],
            []
        )

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in, tensor_scalars])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        output_tensor = tensor_out.data().reshape(x.shape)
        del tensor_in, tensor_out, tensor_scalars
        return [output_tensor]
