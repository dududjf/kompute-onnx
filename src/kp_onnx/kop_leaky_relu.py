import numpy as np
import kp
from .shader_utils import compile_source

DEFAULT_ALPHA = float(0.01)

class LeakyReluOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output
        self.shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf   { float in_data[]; };
layout(set=0, binding=1) buffer OutBuf  { float out_data[]; };

layout (constant_id = 0) const float alpha = 0;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    float x = in_data[gid];
    float y = (x >= 0.0) ? x : (alpha * x);
    out_data[gid] = y;
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"LeakyReluOp({device_name})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        # run(x) or run(x, alpha)
        assert len(inputs) >= 1, "LeakyReluOp requires at least the input tensor"

        data = inputs[0].astype(np.float32)
        flat_data = data.reshape(-1)
        alpha = float(inputs[1]) if len(inputs) >= 2 and inputs[1] is not None else DEFAULT_ALPHA

        tensor_input = self.manager.tensor(flat_data)                            # binding 0
        tensor_output = self.manager.tensor(np.empty_like(flat_data))            # binding 1
        tensors = [tensor_input, tensor_output]

        algo = self.manager.algorithm(tensors, self.shader, spec_consts=[alpha])

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice(tensors)) \
                .record(kp.OpAlgoDispatch(algo)) \
                .record(kp.OpTensorSyncLocal([tensor_output])) \
                .eval()

        outputs = [tensor_output.data().reshape(data.shape)]

        del tensor_input, tensor_output
        return outputs
