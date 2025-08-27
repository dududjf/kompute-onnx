import kp
import numpy as np
from pyshader import python2shader, ivec2, Array, i32
from .shader_utils import compile_source


@python2shader
def compute_shader_bitwise_not_int(index=("input", "GlobalInvocationId", ivec2),
                                   in_data=("buffer", 0, Array(i32)),
                                   out_data=("buffer", 1, Array(i32))):
    i = index.x
    out_data[i] = -in_data[i] - 1

_bitwise_not_code_int = compute_shader_bitwise_not_int.to_spirv()


class BitwiseNotOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

        self.shader_float = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InBuf  { float in_data[];  };
layout(set=0, binding=1) buffer OutBuf { float out_data[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    int bits = floatBitsToInt(in_data[i]);
    bits = ~bits;
    out_data[i] = intBitsToFloat(bits);
}
""")

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def run(self, input_array: np.ndarray):
        tensor_shape = input_array.shape
        flat = input_array.reshape(-1)

        if flat.dtype == np.float32:
            tensor_in = self.manager.tensor(flat.astype(np.float32))
            tensor_out = self.manager.tensor(np.empty_like(flat, dtype=np.float32))
            tensors = [tensor_in, tensor_out]
            algo = self.manager.algorithm(tensors, self.shader_float)

        elif flat.dtype == np.int32:
            tensor_in = self.manager.tensor_t(flat.astype(np.int32))
            tensor_out = self.manager.tensor_t(np.empty_like(flat, dtype=np.int32))
            tensors = [tensor_in, tensor_out]
            algo = self.manager.algorithm(tensors, _bitwise_not_code_int)

        else:
            raise TypeError(f"Unsupported dtype {flat.dtype}. Only float32 and int32 supported.")

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        result = tensor_out.data().reshape(tensor_shape)
        del tensor_in, tensor_out
        return [result]

    def fuse(self,input_tensors: list[tuple[kp.Tensor, list[int]]],updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:

        tensor_in, tensor_shape = input_tensors[0]
        size = np.prod(tensor_shape)

        dtype_enum = tensor_in.data_type()

        if dtype_enum == kp.Tensor.TensorDataTypes.eFloat:
            tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
            updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out], self.shader_float))
        elif dtype_enum == kp.Tensor.TensorDataTypes.eInt:
            tensor_out = self.manager.tensor_t(np.zeros(size, dtype=np.int32))
            updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out], _bitwise_not_code_int))
        else:
            raise TypeError(f"Unsupported tensor data type {dtype_enum}. Only float32 and int32 supported.")

        updated_tensors.append(tensor_out)
        return [(tensor_out, tensor_shape)]
