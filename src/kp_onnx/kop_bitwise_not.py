import kp
import numpy as np
from pyshader import python2shader, ivec2, Array, i32


@python2shader
def compute_shader_bitwise_not(index=("input", "GlobalInvocationId", ivec2),
                               in_data=("buffer", 0, Array(i32)),
                               out_data=("buffer", 1, Array(i32))):
    i = index.x
    out_data[i] = -in_data[i] - 1


_bitwise_not_code = compute_shader_bitwise_not.to_spirv()


class BitwiseNotOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"BitwiseNotOp({device_name})"

    def run(self, *inputs):
        tensor_shape = inputs[0].shape
        numpy_in = inputs[0].reshape(-1)
        # 检查输入类型并创建视图
        int_view = numpy_in.view(np.int32)
        # 创建输出缓冲区
        output_array = np.empty_like(int_view)
        # 使用 tensor_t() 创建张量
        tensor_in = self.manager.tensor_t(int_view)
        tensor_out = self.manager.tensor_t(output_array)
        algo = self.manager.algorithm([tensor_in, tensor_out], _bitwise_not_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
            .record(kp.OpAlgoDispatch(algo)) \
            .record(kp.OpTensorSyncLocal([tensor_out])) \
            .eval()
        result = tensor_out.data()
        result = result.view(numpy_in.dtype)
        outputs = [result.reshape(tensor_shape)]
        del tensor_in
        del tensor_out
        return outputs