import kp
import numpy as np
from .shader_utils import compile_source


class SubOp:
    def __init__(self, manager: kp.Manager, input: list[str], output: list[str]):
        self.manager = manager
        self.input = input
        self.output = output
        self.compiled_shader = compile_source('''
#version 450

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) buffer buf_in_tensor_1 { float in_tensor_1[]; };
layout (binding = 1) buffer buf_in_tensor_2 { float in_tensor_2[]; };
layout (binding = 2) buffer buf_out_tensor { float out_tensor[]; };
layout (constant_id = 0) const float size_x_1f = 0;
layout (constant_id = 1) const float size_y_1f = 0;
layout (constant_id = 2) const float size_z_1f = 0;
layout (constant_id = 3) const float size_x_2f = 0;
layout (constant_id = 4) const float size_y_2f = 0;
layout (constant_id = 5) const float size_z_2f = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;
    uint size_x_1 = uint(size_x_1f);
    uint size_y_1 = uint(size_y_1f);
    uint size_z_1 = uint(size_z_1f);
    uint size_x_2 = uint(size_x_2f);
    uint size_y_2 = uint(size_y_2f);
    uint size_z_2 = uint(size_z_2f);
    uint stride_y_1 = size_z_1;
    uint stride_x_1 = size_y_1 * stride_y_1;
    uint stride_y_2 = size_z_2;
    uint stride_x_2 = size_y_2 * stride_y_2;
    uint stride_y = max(size_z_1, size_z_2);
    uint stride_x = max(size_y_1, size_y_2) * stride_y;
    uint x_1 = min(gx, size_x_1 - 1);
    uint y_1 = min(gy, size_y_1 - 1);
    uint z_1 = min(gz, size_z_1 - 1);
    uint x_2 = min(gx, size_x_2 - 1);
    uint y_2 = min(gy, size_y_2 - 1);
    uint z_2 = min(gz, size_z_2 - 1);
    uint p_1 = x_1 * stride_x_1 + y_1 * stride_y_1 + z_1;
    uint p_2 = x_2 * stride_x_2 + y_2 * stride_y_2 + z_2;
    out_tensor[gx * stride_x + gy * stride_y + gz] = in_tensor_1[p_1] - in_tensor_2[p_2];
}''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"SubOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"SubOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "SubOp requires 2 inputs"
        input_1 = inputs[0]
        input_2 = inputs[1]
        if input_1.ndim < input_2.ndim:
            new_shape_1 = [1] * (input_2.ndim - input_1.ndim) + list(input_1.shape)
            new_shape_2 = list(input_2.shape)
        elif input_2.ndim < input_1.ndim:
            new_shape_1 = list(input_1.shape)
            new_shape_2 = [1] * (input_1.ndim - input_2.ndim) + list(input_2.shape)
        else:
            new_shape_1 = list(input_1.shape)
            new_shape_2 = list(input_2.shape)
        input_shape = []
        for i in range(len(new_shape_1)):
            if new_shape_1[i] == 1:
                input_shape.append(new_shape_2[i])
            elif new_shape_2[i] == 1:
                input_shape.append(new_shape_1[i])
            else:
                assert new_shape_1[i] == new_shape_2[i], f"SubOp requires input {i} of the same shape"
                input_shape.append(new_shape_1[i])
        if input_shape[:-2] != new_shape_1[:-2] and not all(e == 1 for e in new_shape_1[:-2]):
            new_shape_1 = input_shape[:-2] + list(new_shape_1[-2:])
            numpy_in_1 = np.broadcast_to(input_1, new_shape_1).reshape(-1).astype(np.float32)
        else:
            numpy_in_1 = input_1.reshape(-1).astype(np.float32)
        if input_shape[:-2] != new_shape_2[:-2] and not all(e == 1 for e in new_shape_2[:-2]):
            new_shape_2 = input_shape[:-2] + list(new_shape_2[-2:])
            numpy_in_2 = np.broadcast_to(input_2, new_shape_2).reshape(-1).astype(np.float32)
        else:
            numpy_in_2 = input_2.reshape(-1).astype(np.float32)
        if len(new_shape_1) == 1:
            size_x_1 = new_shape_1[0]
            size_y_1 = 1
            size_z_1 = 1
            size_x_2 = new_shape_2[0]
            size_y_2 = 1
            size_z_2 = 1
        elif len(new_shape_1) == 2:
            size_x_1 = new_shape_1[0]
            size_y_1 = new_shape_1[1]
            size_z_1 = 1
            size_x_2 = new_shape_2[0]
            size_y_2 = new_shape_2[1]
            size_z_2 = 1
        else:
            size_x_1 = 1
            for e in new_shape_1[:-2]:
                size_x_1 *= e
            size_y_1 = new_shape_1[-2]
            size_z_1 = new_shape_1[-1]
            size_x_2 = 1
            for e in new_shape_2[:-2]:
                size_x_2 *= e
            size_y_2 = new_shape_2[-2]
            size_z_2 = new_shape_2[-1]
        tensor_in_1 = self.manager.tensor(numpy_in_1)
        tensor_in_2 = self.manager.tensor(numpy_in_2)
        size = 1
        for e in input_shape:
            size *= e
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        workgroup = (max(size_x_1, size_x_2), max(size_y_1, size_y_2), max(size_z_1, size_z_2))
        # print(f"{new_shape_1=} {new_shape_2=} {workgroup=}")
        # print([size_x_1, size_y_1, size_z_1, size_x_2, size_y_2, size_z_2])
        algo = self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out],
                                      self.compiled_shader,
                                      workgroup,
                                      [size_x_1, size_y_1, size_z_1, size_x_2, size_y_2, size_z_2],
                                      [])
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_in_1, tensor_in_2])) \
           .record(kp.OpAlgoDispatch(algo)) \
           .record(kp.OpTensorSyncLocal([tensor_out])) \
           .eval()
        output = tensor_out.data()
        outputs = [output.reshape(input_shape)]
        del tensor_in_1
        del tensor_in_2
        del tensor_out
        return outputs
