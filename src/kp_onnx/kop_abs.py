import kp
import numpy as np
from .shader_utils import compile_source


class AbsOp:
    """
    onnx::Abs 的 Kompute 实现（逐元素一元 | float32）
    """

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout (local_size_x = 1) in;

layout (set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    out_buf[idx] = abs(in_buf[idx]);
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()['device_name']
        return f"AbsOp({dev})"

    __str__ = __repr__

    def run(self, inputs):
        x = inputs[0]
        x_flat = x.reshape(-1).astype(np.float32)

        tensor_in = self.manager.tensor(x_flat)
        tensor_out = self.manager.tensor(np.zeros_like(x_flat))

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

        output_tensor = tensor_out.data().reshape(x.shape)

        del tensor_in, tensor_out
        return [output_tensor]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape = input_tensors[0]
        total = np.prod(shape)
        tensor_out = self.manager.tensor(np.zeros(total, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (total, 1, 1)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            [],
            []
        ))
        return [(tensor_out, shape)]