import kp
import numpy as np
from pyshader import python2shader, ivec2, f32, Array
from pyshader.stdlib import abs, step

@python2shader
def compute_shader_equal(index=("input", "GlobalInvocationId", ivec2),
                         in_a=("buffer", 0, Array(f32)),
                         in_b=("buffer", 1, Array(f32)),
                         out_data=("buffer", 2, Array(f32))):
    i = index.x
    # 绝对误差
    diff = abs(in_a[i] - in_b[i])
    out_data[i] = step(diff, 0.0)

_equal_code = compute_shader_equal.to_spirv()


class EqualOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"EqualOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"EqualOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "EqualOp requires two inputs"
        tensor_shape = inputs[0].shape
        numpy_a = inputs[0].reshape(-1).astype(np.float32)
        numpy_b = inputs[1].reshape(-1).astype(np.float32)
        tensor_a = self.manager.tensor(numpy_a)
        tensor_b = self.manager.tensor(numpy_b)
        tensor_out = self.manager.tensor(np.zeros_like(numpy_a))
        algo = self.manager.algorithm([tensor_a, tensor_b, tensor_out], _equal_code)
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_a, tensor_b])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        outputs = [tensor_out.data().reshape(tensor_shape)]
        del tensor_a
        del tensor_b
        del tensor_out
        # 将浮点数输出转换为布尔值
        outputs[0] = (outputs[0] > 0.5).astype(np.bool_)
        return outputs