import kp
import numpy as np
from pyshader import python2shader, ivec2, Array, i32


def floatBitsToInt(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x.astype(np.float32))
    return np.frombuffer(x.tobytes(), dtype=np.int32).reshape(x.shape)

def intBitsToFloat(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x.astype(np.int32))
    return np.frombuffer(x.tobytes(), dtype=np.float32).reshape(x.shape)


@python2shader
def compute_shader_bitwise_not(index=("input", "GlobalInvocationId", ivec2),
                               in_data=("buffer", 0, Array(i32)),
                               out_data=("buffer", 1, Array(i32))):
    i = index.x
    out_data[i] = -in_data[i] - 1


_bitwise_not_code = compute_shader_bitwise_not.to_spirv()


class BitwiseNotOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def run(self, *inputs):
        tensor_shape = inputs[0].shape
        numpy_in = inputs[0].reshape(-1)
        if np.issubdtype(numpy_in.dtype, np.floating):
            int_data = floatBitsToInt(numpy_in.astype(np.float32))
        elif np.issubdtype(numpy_in.dtype, np.integer):
            int_data = numpy_in.astype(np.int32)
        else:
            raise TypeError(f"Unsupported dtype: {numpy_in.dtype}")
        output_int = np.empty_like(int_data)
        tensor_in = self.manager.tensor_t(int_data)
        tensor_out = self.manager.tensor_t(output_int)
        algo = self.manager.algorithm([tensor_in, tensor_out], _bitwise_not_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        if np.issubdtype(numpy_in.dtype, np.floating):
            result = intBitsToFloat(tensor_out.data())
        else:
            result = tensor_out.data().astype(numpy_in.dtype)
        outputs = [result.reshape(tensor_shape)]
        del tensor_in
        del tensor_out
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]],updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in = input_tensors[0][0]
        tensor_shape = input_tensors[0][1]
        size = np.prod(tensor_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.int32))
        updated_tensors.append(tensor_out)
        updated_algorithms.append(self.manager.algorithm([tensor_in, tensor_out], _bitwise_not_code))
        return [(tensor_out, tensor_shape)]
