import kp
import numpy as np
from .shader_utils import compile_source


class TransposeOp:

    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source(r"""
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

layout(constant_id = 0) const float in_stride_y = 0;
layout(constant_id = 1) const float in_stride_z = 0;
layout(constant_id = 2) const float out_stride_y = 0;
layout(constant_id = 3) const float out_stride_z = 0;

layout(binding=0) readonly  buffer InBuf  { float in_buf[];  };
layout(binding=1) writeonly buffer OutBuf { float out_buf[]; };
layout(binding=2) readonly buffer Params { uint params[]; };

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint in_stride_y = uint(in_stride_y);
    uint in_stride_z = uint(in_stride_z);
    uint out_stride_y = uint(out_stride_y);
    uint out_stride_z = uint(out_stride_z);

    uint base_in  = params[2u*gx + 0u];
    uint base_out = params[2u*gx + 1u];

    uint in_index  = base_in  + gy * in_stride_y  + gz * in_stride_z;
    uint out_index = base_out + gy * out_stride_z + gz * out_stride_y;

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
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
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
        ndim = len(shape_in)

        if ndim <= 1:
            return [(tensor_in, shape_in)]

        if len(input_tensors) > 1:
            perm_vals = input_tensors[1][0].data().astype(np.int64).tolist()
            perm_vals = [p % ndim for p in perm_vals]
            assert len(perm_vals) == ndim, \
                f"Inconsistent permutation {perm_vals!r} with shape {shape_in!r}."
        else:
            perm_vals = list(reversed(range(ndim)))

        out_shape = [int(shape_in[i]) for i in perm_vals]
        num_elems_final = int(np.prod(out_shape))
        tensor_out = self.manager.tensor(np.zeros(num_elems_final, dtype=np.float32))
        updated_tensors.append(tensor_out)

        def _perm_to_steps(perm):
            n = len(perm)
            curr = list(range(n))
            steps = []
            for i in range(n):
                want = perm[i]
                if curr[i] == want:
                    continue
                j = curr.index(want)
                steps.append((i, j))
                curr[i], curr[j] = curr[j], curr[i]
            return steps

        swaps = _perm_to_steps(perm_vals)
        if len(swaps) > 1:
            num_elems_curr = int(np.prod(shape_in))
            tensor_tmp0 = self.manager.tensor(np.zeros(num_elems_curr, dtype=np.float32))
            tensor_tmp1 = self.manager.tensor(np.zeros(num_elems_curr, dtype=np.float32))
            updated_tensors.extend([tensor_tmp0, tensor_tmp1])
        else:
            tensor_tmp0 = tensor_tmp1 = None

        tensor_curr = tensor_in

        def _compute_strides(shape):
            strides = [1] * len(shape)
            acc = 1
            for i in range(len(shape) - 1, -1, -1):
                strides[i] = acc
                acc *= int(shape[i])
            return strides

        for si, (axis_a, axis_b) in enumerate(swaps):
            in_strides = _compute_strides(shape_in)
            shape_next = list(shape_in)
            shape_next[axis_a], shape_next[axis_b] = shape_next[axis_b], shape_next[axis_a]
            out_strides = _compute_strides(shape_next)

            size_a, size_b = int(shape_in[axis_a]), int(shape_in[axis_b])

            batch_axes = [d for d in range(ndim) if d not in (axis_a, axis_b)]
            batch = int(np.prod([shape_in[d] for d in batch_axes])) if batch_axes else 1

            params = np.zeros(batch * 2, dtype=np.uint32)
            for b in range(batch):
                q, base_in, base_out = b, 0, 0
                for d in reversed(batch_axes):
                    dim = int(shape_in[d])
                    c = q % dim
                    q //= dim
                    base_in += c * int(in_strides[d])
                    base_out += c * int(out_strides[d])
                params[2 * b], params[2 * b + 1] = base_in, base_out

            spec_consts = [
                in_strides[axis_a],
                in_strides[axis_b],
                out_strides[axis_a],
                out_strides[axis_b],
            ]

            t_params = self.manager.tensor_t(params, kp.TensorTypes.device)
            updated_tensors.append(t_params)

            if si == len(swaps) - 1 or tensor_tmp0 is None:
                tensor_next = tensor_out
            else:
                tensor_next = tensor_tmp1 if tensor_curr is tensor_tmp0 else tensor_tmp0

            updated_algorithms.append(self.manager.algorithm(
                [tensor_curr, tensor_next, t_params],
                self.shader,
                [batch, size_a, size_b],
                spec_consts,
                []
            ))

            shape_in, tensor_curr = shape_next, tensor_next

        return [(tensor_out, out_shape)]