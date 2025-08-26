import numpy as np
import kp
from .shader_utils import compile_source

DEFAULT_LAMBDA = float(0.5)
DEFAULT_BIAS = float(0.0)


class ShrinkOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf     { float in_data[];  };
layout(set=0, binding=1) buffer OutBuf    { float out_data[]; };

layout(constant_id=0) const float Lambda = 0;
layout(constant_id=1) const float Bias = 0;

void main() {
    uint i = gl_GlobalInvocationID.x;
    float x = in_data[i];
    float y = 0.0;
    y = (x > Lambda) ? (x - Bias) : (x < -Lambda) ? (x + Bias) : 0.0;
    out_data[i] = y;
}
""")

    def __repr__(self):
        return f"ShrinkOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        data = inputs[0].astype(np.float32)
        flat_data = data.reshape(-1)

        lam_val = float(inputs[1]) if len(inputs) > 1 and inputs[1] is not None else DEFAULT_LAMBDA
        bias_val = float(inputs[2]) if len(inputs) > 2 and inputs[2] is not None else DEFAULT_BIAS

        tensor_in = self.manager.tensor(flat_data)
        tensor_out = self.manager.tensor(np.empty_like(flat_data))
        tensors = [tensor_in, tensor_out]

        algo = self.manager.algorithm(tensors, self.shader, spec_consts=[lam_val, bias_val])
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
            .record(kp.OpAlgoDispatch(algo)) \
            .record(kp.OpTensorSyncLocal([tensor_out])) \
            .eval()

        outputs = [tensor_out.data().reshape(data.shape)]
        del tensor_in, tensor_out
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in = input_tensors[0][0]
        tensor_shape = input_tensors[0][1]
        size = np.prod(tensor_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))

        lam_val = float(input_tensors[1][0]) \
            if len(input_tensors) >= 2 and input_tensors[1][0] is not None else DEFAULT_LAMBDA
        bias_val = float(input_tensors[2][0]) \
            if len(input_tensors) >= 3 and input_tensors[2][0] is not None else DEFAULT_BIAS

        updated_tensors.append(tensor_out)
        updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out],
                                                         self.shader, spec_consts=[lam_val, bias_val]))
        return [(tensor_out, tensor_shape)]
