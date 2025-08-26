import kp
import numpy as np
from .shader_utils import compile_source

DEFAULT_ALPHA = 1.0


class CeluOp:
    """
    onnx::Celu 的 Kompute 实现（逐元素一元 | float32）
    """

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout (local_size_x = 1) in;

layout (set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout (constant_id = 0) const float alpha = 0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
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
        alpha = float(inputs[1]) if len(inputs) > 1 else DEFAULT_ALPHA
        x_flat = x.reshape(-1).astype(np.float32)

        tensor_in = self.manager.tensor(x_flat)
        tensor_out = self.manager.tensor(np.zeros_like(x_flat))

        workgroup = (x_flat.size, 1, 1)
        algo = self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            [alpha],
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

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape = input_tensors[0]
        alpha = float(input_tensors[1][0].data()) if len(input_tensors) > 1 else DEFAULT_ALPHA
        total = np.prod(shape)
        tensor_out = self.manager.tensor(np.zeros(total, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (total, 1, 1)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            [alpha],
            []
        ))
        return [(tensor_out, shape)]