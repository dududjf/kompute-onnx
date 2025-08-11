import kp
import numpy as np
from pyshader import python2shader, ivec2, f32, Array
from pyshader.stdlib import step


@python2shader
def compute_shader_not(index=("input", "GlobalInvocationId", ivec2),
                       in_data=("buffer", 0, Array(f32)),
                       out_data=("buffer", 1, Array(f32))):
    i = index.x
    #绝对值
    abs_val = abs(in_data[i])
    epsilon = 1e-8
    #通过 step(epsilon, abs_val) 判断是否大于 epsilon（1e-8），得到 0 或 1
    #用 1.0 - step(...) 等价于逻辑非
    is_zero = 1.0 - step(epsilon, abs_val)
    out_data[i] = is_zero

_not_code = compute_shader_not.to_spirv()


class NotOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"NotOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"NotOp({device_name})"

    def run(self, *inputs):
        tensor_shape = inputs[0].shape
        numpy_in = inputs[0].reshape(-1).astype(np.float32)
        tensor_in = self.manager.tensor(numpy_in)
        tensor_out = self.manager.tensor(np.zeros_like(numpy_in))
        algo = self.manager.algorithm([tensor_in, tensor_out], _not_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        outputs = [tensor_out.data().reshape(tensor_shape)]
        del tensor_in
        del tensor_out
        # 将浮点数输出转换为布尔值
        outputs[0] = (outputs[0] > 0.5).astype(np.bool_)
        return outputs 