import kp
import numpy as np
from .shader_utils import compile_source


class TransposeOp:

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

layout(binding=0) readonly  buffer InBuf     { float in_buf[];     };
layout(binding=1) writeonly buffer OutBuf    { float out_buf[];    };
layout(binding=2) readonly  buffer OutShape  { uint  out_shape[];  };
layout(binding=3) readonly  buffer InStrides { uint  in_strides[]; };
layout(binding=4) readonly  buffer Perm      { uint  perm[];       };

void main() {
    uint ndim = out_shape.length();

    uint cols = out_shape[ndim - 1u];
    uint rows = (ndim >= 2u) ? out_shape[ndim - 2u] : 1u;

    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint in_index = 0u;

    if (ndim >= 3u) {
        int tmp = int(gz);
        for (int i = int(ndim) - 3; i >= 0; --i) {
            int dim_i    = int(out_shape[i]);
            int coord_i  = tmp % dim_i;   
            tmp /= int(dim_i);
            uint pin     = perm[i];
            in_index    += coord_i * in_strides[pin];
        }
    }

    if (ndim >= 2u) {
        uint pin = perm[ndim - 2u];
        in_index += gy * in_strides[pin];
    }

    {
        uint pin = perm[ndim - 1u];
        in_index += gx * in_strides[pin];
    }

    uint out_index = gx;
    if (ndim >= 2u) out_index += gy * cols;
    if (ndim >= 3u) out_index += gz * (rows * cols);

    out_buf[out_index] = in_buf[in_index];
}
""")

    def __repr__(self):
        dev = self.manager.get_device_properties()["device_name"]
        return f"TransposeOp({dev})"

    __str__ = __repr__

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32) \
                if isinstance(inp, np.ndarray) else np.array(inp, dtype=np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape) if isinstance(inp, np.ndarray) else []))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(
            input_tensors, updated_algorithms, updated_tensors
        )
        tensor_out, output_shape = output_tensor_and_shape[0]

        tensors_to_device = [t[0] for t in input_tensors] + updated_tensors

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice(tensors_to_device))
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
        ndim_in = len(shape_in)

        if len(input_tensors) > 1 and input_tensors[1][0].data().size > 0:
            perm_vals = input_tensors[1][0].data().reshape(-1).astype(np.int64).tolist()
            perm_vals = [p % ndim_in for p in perm_vals]
        else:
            perm_vals = list(reversed(range(ndim_in)))
        assert len(perm_vals) == ndim_in, \
            f"Inconsistent permutation {perm_vals!r} with shape {shape_in!r}."

        def _compute_strides(shape):
            strides = [1] * len(shape)
            acc = 1
            for i in range(len(shape) - 1, -1, -1):
                strides[i] = acc
                acc *= int(shape[i])
            return strides

        in_strides = _compute_strides(shape_in)

        out_shape = [int(shape_in[i]) for i in perm_vals]
        num_elems = int(np.prod(out_shape))
        tensor_out = self.manager.tensor(np.zeros(num_elems, dtype=np.float32))
        updated_tensors.append(tensor_out)

        t_out_shape = self.manager.tensor_t(np.array(out_shape, dtype=np.uint32), tensor_type=kp.TensorTypes.device)
        t_in_strides = self.manager.tensor_t(np.array(in_strides, dtype=np.uint32), tensor_type=kp.TensorTypes.device)
        t_perm = self.manager.tensor_t(np.array(perm_vals, dtype=np.uint32), tensor_type=kp.TensorTypes.device)
        updated_tensors.extend([t_out_shape, t_in_strides, t_perm])

        if ndim_in == 1:
            grid = (out_shape[0], 1, 1)
        else:
            cols = out_shape[-1]
            rows = out_shape[-2]
            batch = int(np.prod(out_shape[:-2])) if ndim_in > 2 else 1
            grid = (cols, rows, batch)

        updated_algorithms.append(
            self.manager.algorithm(
                [tensor_in, tensor_out, t_out_shape, t_in_strides, t_perm],
                self.shader,
                grid,
                [],
                []
            )
        )

        return [(tensor_out, out_shape)]