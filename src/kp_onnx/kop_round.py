import numpy as np
import kp
from pyshader import python2shader, f32, ivec2, Array
from pyshader.stdlib import round_even


@python2shader
def compute_shader_round(index=("input", "GlobalInvocationId", ivec2),
                         in_data=("buffer", 0, Array(f32)),
                         out_data=("buffer", 1, Array(f32)),
):
    i = index.x
    out_data[i] = round_even(in_data[i])


class RoundOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output
        self.shader = compute_shader_round.to_spirv()

    def __repr__(self):
        return f"RoundOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        data = inputs[0].astype(np.float32)

        tensor_in = self.manager.tensor(data)                  # binding 0
        tensor_out = self.manager.tensor(np.empty_like(data))  # binding 1

        algo = self.manager.algorithm([tensor_in, tensor_out], self.shader)

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        outputs = [tensor_out.data().reshape(data.shape)]
        del tensor_in, tensor_out
        return outputs
