import numpy as np
import kp
from .shader_utils import compile_source


class UniqueOp:
    def __init__(self, manager: kp.Manager, axis=None, sorted=1):
        self.manager = manager
        self.axis = axis
        self.sorted = sorted
        self.compiled_shader = compile_source("""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;

layout(binding = 0) readonly   buffer in_buf                { float in_tensor[];     };
layout(binding = 1) writeonly  buffer out_buf               { float out_tensor[];    };
layout(binding = 2) writeonly  buffer indice_buf            { float indice_tensor[];     };
layout(binding = 3) writeonly  buffer inverse_indices_buf   { float inverse_indices _tensor[];  };
layout(binding = 4) writeonly  buffer in_buf                { float counts _tensor[];     };

""")

    def __repr__(self):
        return f"UniqueOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                    if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else [len(inp)]))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        axis = self.axis
        sorted = self.sorted

        if axis is None:
            pass
        output_tensors_and_shapes = []



        return output_tensors_and_shapes