import kp
import numpy as np
from .shader_utils import compile_source


class CeluOp:
    """
    onnx::Celu 的 Kompute 实现（逐元素一元 | float32）
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
    float x = in_buf[idx];
    // CELU(x) = x (x>=0) ; alpha * (exp(x/alpha) - 1) (x<0)
    out_buf[idx] = (x >= 0.0) ? x : (alpha * (exp(x / alpha) - 1.0));
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"CeluOp({dev})"

    __str__ = __repr__

    def run(self, inputs):
        x = inputs[0]
        alpha = inputs[1] if len(inputs) > 1 else 1.0

        x_flat = x.astype(np.float32).reshape(-1)

        tensor_in = self.manager.tensor(x_flat)
        tensor_out = self.manager.tensor(np.zeros_like(x_flat))
        tensor_scalars = self.manager.tensor(np.array([alpha], dtype=np.float32))

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