import numpy as np
import kp
from .shader_utils import compile_source


class LeakyReluOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        self._local_size_x = None
        self.shader = None
        self.shader_tmpl = """
#version 450
layout(local_size_x = {LOCAL_SIZE_X}, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf   {{ float data[];     }};
layout(set=0, binding=1) buffer OutBuf  {{ float out_data[]; }};
layout(set=0, binding=2) buffer Scalars {{ float scalars[];  }}; // [alpha]
layout(set=0, binding=3) buffer IntsBuf {{ int   ints[];     }}; // [size]

void main() {{
    uint gid = gl_GlobalInvocationID.x;
    int size = ints[0];
    if (gid >= uint(size)) return;

    float x = data[gid];
    float a = scalars[0];
    // y = x if x>=0 else a*x
    float y = (x >= 0.0) ? x : (a * x);
    out_data[gid] = y;
}}
"""

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"LeakyReluOp({device_name})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        # run(x) or run(x, alpha)
        if len(inputs) < 1:
            raise ValueError("LeakyReluOp requires at least the input tensor")

        data = np.asarray(inputs[0], dtype=np.float32)
        flat_data = data.reshape(-1).astype(np.float32)
        data_size = int(flat_data.size)

        alpha = 0.01
        if len(inputs) >= 2 and inputs[1] is not None:
            alpha = float(inputs[1])

        # Dynamic selection of local_size_x: cap = min(max_inv, max_x, data_size), take the largest power of 2 ≤ cap
        props = self.manager.get_device_properties()
        max_inv = int(props["max_work_group_invocations"])
        max_x = int(props["max_work_group_size"][0])
        lx_cap = max(1, min(max_inv, max_x, data_size))
        local_size_x = 1
        while (local_size_x << 1) <= lx_cap:
            local_size_x <<= 1

        if (self.shader is None) or (self._local_size_x != local_size_x):
            self._local_size_x = local_size_x
            shader_code = self.shader_tmpl.format(LOCAL_SIZE_X=local_size_x)
            self.shader = compile_source(shader_code)

        tensor_input = self.manager.tensor(flat_data)                           # binding 0
        tensor_output = self.manager.tensor(np.empty_like(flat_data))            # binding 1
        tensor_scalars = self.manager.tensor(np.asarray([alpha], np.float32))     # binding 2
        tensor_ints = self.manager.tensor(np.asarray([data_size], np.int32))   # binding 3
        tensors = [tensor_input, tensor_output, tensor_scalars, tensor_ints]

        groups_x = (data_size + local_size_x - 1) // local_size_x
        workgroup = (int(groups_x), 1, 1)
        algorithm = self.manager.algorithm(tensors, self.shader, workgroup)

        sequence = self.manager.sequence()
        sequence.record(kp.OpTensorSyncDevice(tensors)) \
                .record(kp.OpAlgoDispatch(algorithm)) \
                .record(kp.OpTensorSyncLocal([tensor_output])) \
                .eval()

        output = [tensor_output.data().reshape(data.shape)]

        del tensor_input, tensor_output, tensor_scalars, tensor_ints, algorithm, sequence
        return output
