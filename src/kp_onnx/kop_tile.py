# kp_onnx/kop_tile.py
import numpy as np
import kp
from .shader_utils import compile_source


class TileOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.shader = compile_source('''
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) buffer buf_in  { float in_data[];  };
layout (binding = 1) buffer buf_out { float out_data[]; };

layout (constant_id = 0) const float size_x_in_f  = 0;
layout (constant_id = 1) const float size_y_in_f  = 0;
layout (constant_id = 2) const float size_z_in_f  = 0;
layout (constant_id = 3) const float size_x_out_f = 0;
layout (constant_id = 4) const float size_y_out_f = 0;
layout (constant_id = 5) const float size_z_out_f = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint size_x_in  = uint(size_x_in_f);
    uint size_y_in  = uint(size_y_in_f);
    uint size_z_in  = uint(size_z_in_f);
    uint size_x_out = uint(size_x_out_f);
    uint size_y_out = uint(size_y_out_f);
    uint size_z_out = uint(size_z_out_f);

    uint stride_y_in  = size_z_in;
    uint stride_x_in  = size_y_in * stride_y_in;
    uint stride_y_out = size_z_out;
    uint stride_x_out = size_y_out * stride_y_out;

    // 输出 -> 输入：按输入维度取模
    uint ix = (size_x_in  == 0u) ? 0u : (gx % size_x_in);
    uint iy = (size_y_in  == 0u) ? 0u : (gy % size_y_in);
    uint iz = (size_z_in  == 0u) ? 0u : (gz % size_z_in);

    uint p_in  = ix * stride_x_in  + iy * stride_y_in  + iz;
    uint p_out = gx * stride_x_out + gy * stride_y_out + gz;

    out_data[p_out] = in_data[p_in];
}
''')

    def __repr__(self):
        return f"TileOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _sizes3(shape):
        if len(shape) == 0:
            return 1, 1, 1
        if len(shape) == 1:
            return shape[0], 1, 1
        if len(shape) == 2:
            return shape[0], shape[1], 1
        x = np.prod(shape[:-2])
        y = shape[-2]
        z = shape[-1]
        return x, y, z

    def run(self, *inputs):
        assert len(inputs) >= 2, "TileOp needs (x, repeats)"
        x = inputs[0]
        repeats = inputs[1].astype(np.float32)
        assert repeats.ndim == 1, "repeats must be 1-D"
        assert repeats.size == x.ndim, "len(repeats) must equal x.ndim"

        in_dims = list(x.shape)
        out_dims = [int(in_dims[d]) * int(repeats[d]) for d in range(x.ndim)]

        sx_in,  sy_in,  sz_in = self._sizes3(in_dims)
        sx_out, sy_out, sz_out = self._sizes3(out_dims)

        tensor_in = self.manager.tensor(x.reshape(-1))
        size = np.prod(out_dims)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))

        workgroup = (sx_out, sy_out, sz_out)

        algo = self.manager.algorithm([tensor_in, tensor_out],
                                      self.shader, workgroup,
                                      [sx_in, sy_in, sz_in, sx_out, sy_out, sz_out],
                                      [])
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()

        outputs = [tensor_out.data().reshape(out_dims)]

        del tensor_in, tensor_out, algo, seq
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert len(input_tensors) >= 2, "TileOp.fuse needs (x_tensor, repeats_tensor)"
        x_tensor = input_tensors[0][0]
        x_dims = input_tensors[0][1]
        reps_tensor = input_tensors[1][0]

        assert reps_tensor.size == len(x_dims), "len(repeats) must equal rank"

        output_shape = [int(x_dims[d]) * int(reps_tensor[d]) for d in range(len(x_dims))]

        sx_in,  sy_in,  sz_in = self._sizes3(x_dims)
        sx_out, sy_out, sz_out = self._sizes3(output_shape)

        out_size = np.prod(output_shape) if output_shape else 1
        tensor_out = self.manager.tensor(np.empty(out_size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (sx_out, sy_out, sz_out)

        updated_algorithms.append(self.manager.algorithm([x_tensor, tensor_out],
                                                         self.shader,
                                                         workgroup,
                                                         [sx_in, sy_in, sz_in, sx_out, sy_out, sz_out],
                                                         []))

        return [(tensor_out, output_shape)]
