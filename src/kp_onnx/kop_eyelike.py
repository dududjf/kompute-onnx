import kp
import numpy as np
from .shader_utils import compile_source

DEFAULT_K = 0


class EyeLikeOp:

    def __init__(self, manager: kp.Manager):
        self.k_val = DEFAULT_K
        self.manager = manager
        self.shader = compile_source(r"""
#version 450
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0) writeonly buffer Out { float out_buf[]; };

layout(constant_id = 0) const float ROWS_F = 0.0;
layout(constant_id = 1) const float COLS_F = 0.0;
layout(constant_id = 2) const float K_F    = 0.0;

void main() {
    uint rows = uint(ROWS_F);
    uint cols = uint(COLS_F);
    int  k    = int(K_F);

    uint r = gl_GlobalInvocationID.y;
    uint c = gl_GlobalInvocationID.x;

    out_buf[r * cols + c] = ((int(c) - int(r)) == k) ? 1.0 : 0.0;
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()["device_name"]
        return f"EyeLikeOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        numpy_in = inputs[0].reshape(-1).astype(np.float32) \
            if isinstance(inputs[0], np.ndarray) else np.array(inputs[0], dtype=np.float32)
        tensor = self.manager.tensor(numpy_in)
        input_tensors.append((tensor, list(inputs[0].shape) if isinstance(inputs[0], np.ndarray) else []))
        if len(inputs) > 1:
            numpy_in = np.array(inputs[1], dtype=np.int32).reshape(-1)
            tensor = self.manager.tensor_t(numpy_in, kp.TensorTypes.device)
            input_tensors.append((tensor, list(numpy_in.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
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
        shape_in = input_tensors[0][1]
        if len(shape_in) == 1:
            rows, cols = shape_in[0], shape_in[0]
            out_shape = [rows, cols]
        elif len(shape_in) == 2:
            rows, cols = shape_in
            out_shape = [rows, cols]
        else:
            raise AssertionError(f"EyeLike only accepts 1D or 2D input, got {shape_in}")

        self.k_val = float(input_tensors[1][0].data()) if len(input_tensors) > 1 else self.k_val

        size = rows * cols
        shader = self.shader
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (cols, rows, 1)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_out],
            shader,
            workgroup,
            [rows, cols, self.k_val],
            []
        ))

        return [(tensor_out, out_shape)]