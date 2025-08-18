import numpy as np
import kp
from pyshader import python2shader, ivec2, f32, Array
from pyshader.stdlib import floor


@python2shader
def compute_shader_floor(index=("input", "GlobalInvocationId", ivec2),
                         in_data=("buffer", 0, Array(f32)),
                         out_data=("buffer", 1, Array(f32))):
    i = index.x
    out_data[i] = floor(in_data[i])


class FloorOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output
        self.shader = compute_shader_floor.to_spirv()

    def __repr__(self):
        return f"FloorOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        assert len(inputs) >= 1, "FloorOp needs the input tensor"

        data = inputs[0].astype(np.float32)
        flat_data = data.reshape(-1)

        tensor_in = self.manager.tensor(flat_data)                  # binding 0
        tensor_out = self.manager.tensor(np.empty_like(data))       # binding 1

        algo = self.manager.algorithm([tensor_in, tensor_out], self.shader)

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        outputs = [tensor_out.data().reshape(data.shape)]
        del tensor_in, tensor_out, algo, seq
        return outputs
