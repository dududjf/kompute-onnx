import kp
import numpy as np
from .shader_utils import compile_source

# Selu 默认常量
DEFAULT_ALPHA = float(1.6732631921768188)
DEFAULT_GAMMA = float(1.0507009873554805)


class SeluOp:
    """
    onnx::Selu 的 Kompute 实现（逐元素一元 | float32）
    """

    def __init__(self, manager: kp.Manager):
        self.alpha = DEFAULT_ALPHA
        self.gamma = DEFAULT_GAMMA
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout (local_size_x = 1) in;

layout (set = 0, binding = 0) readonly buffer InBuf  { float in_buf[]; };
layout (set = 0, binding = 1) writeonly buffer OutBuf { float out_buf[]; };

layout (constant_id = 0) const float alpha = 0;
layout (constant_id = 1) const float gamma = 0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float x = in_buf[idx];
    // SELU(x) = gamma * ( x (x>=0) ; alpha*(exp(x)-1) (x<0) )
    float y = x >= 0.0 ? x : alpha * (exp(x) - 1.0);
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
        #  指定alpha、gamma值，或使用 SELU 默认常量
        alpha = float(inputs[1]) if len(inputs) > 1 else DEFAULT_ALPHA
        gamma = float(inputs[2]) if len(inputs) > 2 else DEFAULT_GAMMA
        x_flat = x.reshape(-1).astype(np.float32)

        tensor_in = self.manager.tensor(x_flat)
        tensor_out = self.manager.tensor(np.zeros_like(x_flat))

        workgroup = (x_flat.size, 1, 1)
        algo = self.manager.algorithm(
            [tensor_in, tensor_out],
            self.shader,
            workgroup,
            [alpha, gamma],
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
            [self.alpha, self.gamma],
            []
        ))
        return [(tensor_out, shape)]
