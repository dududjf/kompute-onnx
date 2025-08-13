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

layout(set=0, binding=0) buffer InBuf   { float data[];     };
layout(set=0, binding=1) buffer OutBuf  { float out_data[]; };
layout(set=0, binding=2) buffer MinBuf  { float min_value[];};

void main() {
    uint i = gl_GlobalInvocationID.x;
    float v = data[i];
    float m = min_value[0];
    out_data[i] = (v < m) ? m : v;
}
""")

        self.shader_max = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf   { float data[];     };
layout(set=0, binding=1) buffer OutBuf  { float out_data[]; };
layout(set=0, binding=2) buffer MaxBuf  { float max_value[];};

void main() {
    uint i = gl_GlobalInvocationID.x;
    float v = data[i];
    float M = max_value[0];
    out_data[i] = (v > M) ? M : v;
}
""")

    def __repr__(self):
        return f"ClipOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        # run(x [, min_scalar] [, max_scalar])
        assert len(inputs) >= 1, "ClipOp needs at least the input tensor"

        data = np.asarray(inputs[0], dtype=np.float32)
        flat_data = data.reshape(-1)

        min_scalar = inputs[1] if len(inputs) > 1 else None
        max_scalar = inputs[2] if len(inputs) > 2 else None

        if min_scalar is not None:
            min_arr = np.asarray(min_scalar, dtype=np.float32).reshape(-1)
            assert min_arr.size == 1, "min must be scalar (empty-shape / size==1)"
            min_val = np.asarray(min_arr, dtype=np.float32)
        else:
            min_val = None

        if max_scalar is not None:
            max_arr = np.asarray(max_scalar, dtype=np.float32).reshape(-1)
            assert max_arr.size == 1, "max must be scalar (empty-shape / size==1)"
            max_val = np.asarray(max_arr, dtype=np.float32)
        else:
            max_val = None

        # case 1
        if (min_val is None) and (max_val is None):
            outputs = [data]
            return outputs

        # case 2：only min
        if (min_val is not None) and (max_val is None):
            tensor_in = self.manager.tensor(flat_data)
            tensor_out = self.manager.tensor(np.empty_like(flat_data))
            tensor_min = self.manager.tensor(min_val)
            tensors = [tensor_in, tensor_out, tensor_min]

            algo = self.manager.algorithm(tensors, self.shader_min)
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice(tensors)) \
               .record(kp.OpAlgoDispatch(algo)) \
               .record(kp.OpTensorSyncLocal([tensor_out])) \
               .eval()

            outputs = [tensor_out.data().reshape(data.shape)]
            del tensor_in, tensor_out, tensor_min, algo, seq
            return outputs

        # case 3：only max
        if (min_val is None) and (max_val is not None):
            tensor_in = self.manager.tensor(flat_data)
            tensor_out = self.manager.tensor(np.empty_like(flat_data))
            tensor_max = self.manager.tensor(max_val)
            tensors = [tensor_in, tensor_out, tensor_max]

            algo = self.manager.algorithm(tensors, self.shader_max)
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice(tensors)) \
               .record(kp.OpAlgoDispatch(algo)) \
               .record(kp.OpTensorSyncLocal([tensor_out])) \
               .eval()

            outputs = [tensor_out.data().reshape(data.shape)]
            del tensor_in, tensor_out, tensor_max, algo, seq
            return outputs

        # case 4
        tensor_in = self.manager.tensor(flat_data)
        tensor_tmp = self.manager.tensor(np.empty_like(flat_data))
        tensor_out = self.manager.tensor(np.empty_like(flat_data))
        tensor_min = self.manager.tensor(min_val)
        tensor_max = self.manager.tensor(max_val)

        # 第一次：min
        algo_min = self.manager.algorithm([tensor_in,  tensor_tmp, tensor_min], self.shader_min)
        # 第二次：max
        algo_max = self.manager.algorithm([tensor_tmp, tensor_out, tensor_max], self.shader_max)

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in, tensor_min, tensor_max])) \
           .record(kp.OpAlgoDispatch(algo_min)) \
           .record(kp.OpAlgoDispatch(algo_max)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        outputs = [tensor_out.data().reshape(data.shape)]
        del tensor_in, tensor_tmp, tensor_out, tensor_min, tensor_max, algo_min, algo_max, seq
        return outputs
