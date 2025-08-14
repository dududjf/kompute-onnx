import numpy as np
import kp
from .shader_utils import compile_source


class EluOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        self.local_size_x = None
        self.shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf     { float in_data[];    };
layout(set=0, binding=1) buffer OutBuf    { float out_data[]; };
layout(set=0, binding=2) buffer Scalars   { float scalars[]; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    float x = in_data[gid];
    float alpha = scalars[0];
    float y = (x > 0.0) ? x : alpha * (exp(x) - 1.0);
    out_data[gid] = y;
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"EluOp({device_name})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        # run(x) or run(x, alpha)
        assert len(inputs) >= 1, "EluOp requires at least the input tensor"

        data = inputs[0].astype(np.float32)
        flat_data = data.reshape(-1)
        alpha = np.asarray([inputs[1] if len(inputs) >= 2 and inputs[1] is not None else 1.0], dtype=np.float32)

        tensor_input = self.manager.tensor(flat_data)                          # binding 0
        tensor_output = self.manager.tensor(np.empty_like(flat_data))           # binding 1
        tensor_scalars = self.manager.tensor(alpha)                             # binding 2
        tensors = [tensor_input, tensor_output, tensor_scalars]

        algo = self.manager.algorithm(tensors, self.shader)

        sequence = self.manager.sequence()
        sequence.record(kp.OpTensorSyncDevice(tensors)) \
                .record(kp.OpAlgoDispatch(algo)) \
                .record(kp.OpTensorSyncLocal([tensor_output])) \
                .eval()

        outputs = [tensor_output.data().reshape(data.shape)]

        del tensor_input, tensor_output, tensor_scalars,algo, sequence
        return outputs
