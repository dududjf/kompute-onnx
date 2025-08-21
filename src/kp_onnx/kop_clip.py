import numpy as np
import kp
from .shader_utils import compile_source


class ClipOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

        self.shader_min = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf   { float in_data[];  };
layout(set=0, binding=1) buffer OutBuf  { float out_data[]; };

layout(constant_id=0) const float min_value = 0;

void main() {
    uint i = gl_GlobalInvocationID.x;
    float v = in_data[i];
    out_data[i] = (v < min_value) ? min_value : v;
}
""")

        self.shader_minmax = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf   { float in_data[];  };
layout(set=0, binding=1) buffer OutBuf  { float out_data[]; };

layout(constant_id=0) const float min_value = 0;
layout(constant_id=1) const float max_value = 0;

void main() {
    uint i = gl_GlobalInvocationID.x;
    float v = in_data[i];

    v = (v < min_value) ? min_value : v;
    v = (v > max_value) ? max_value : v;
    out_data[i] = v;
}
""")

    def __repr__(self):
        return f"ClipOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        data = inputs[0].astype(np.float32)
        flat_data = data.reshape(-1)

        min_scalar = inputs[1] if len(inputs) > 1 else None
        max_scalar = inputs[2] if len(inputs) > 2 else None

        min_val = np.asarray(min_scalar, dtype=np.float32) if min_scalar is not None else None
        if min_val is not None:
            assert min_val.ndim == 0 or min_val.size == 1, "min must be scalar"

        max_val = np.asarray(max_scalar, dtype=np.float32) if max_scalar is not None else None
        if max_val is not None:
            assert max_val.ndim == 0 or max_val.size == 1, "max must be scalar"

        # case 1: min and max are None
        if min_val is None and max_val is None:
            outputs = [data]
            return outputs

        tensor_in = self.manager.tensor(flat_data)
        tensor_out = self.manager.tensor(np.empty_like(flat_data))
        tensors = [tensor_in, tensor_out]
        seq = self.manager.sequence()

        # case 2：only min
        if min_val is not None and max_val is None:
            algo = self.manager.algorithm(tensors, self.shader_min, spec_consts=[min_val])
            seq.record(kp.OpTensorSyncDevice(tensors)) \
               .record(kp.OpAlgoDispatch(algo)) \
               .record(kp.OpTensorSyncLocal([tensor_out])) \
               .eval()

            outputs = [tensor_out.data().reshape(data.shape)]
            del tensor_in, tensor_out
            return outputs

        # case 3: min and max
        algo = self.manager.algorithm(tensors, self.shader_minmax, spec_consts=[min_val, max_val])
        seq.record(kp.OpTensorSyncDevice(tensors)) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        outputs = [tensor_out.data().reshape(data.shape)]
        del tensor_in, tensor_out
        return outputs
