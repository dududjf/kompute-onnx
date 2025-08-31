import kp
import numpy as np
from .shader_utils import compile_source, broadcast_to


class PReLUOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source('''
#version 450

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) buffer buf_in_tensor_data     { float in_tensor_data[]; };
layout (binding = 1) buffer buf_in_tensor_slope { float in_tensor_slope[]; };
layout (binding = 2) buffer buf_out_tensor      { float out_tensor[]; };
layout (constant_id = 0) const float size_x_data = 0;
layout (constant_id = 1) const float size_y_data = 0;
layout (constant_id = 2) const float size_z_data = 0;
layout (constant_id = 3) const float size_x_slope = 0;
layout (constant_id = 4) const float size_y_slope = 0;
layout (constant_id = 5) const float size_z_slope = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint x_data = uint(size_x_data);
    uint y_data = uint(size_y_data);
    uint z_data = uint(size_z_data);
    uint stride_y_data = z_data;
    uint stride_x_data = y_data * stride_y_data;
    uint idx_data = min(gx, x_data - 1) * stride_x_data + min(gy, y_data - 1) * stride_y_data + min(gz, z_data - 1);

    uint x_slope = uint(size_x_slope);
    uint y_slope = uint(size_y_slope);
    uint z_slope = uint(size_z_slope);
    uint stride_y_slope = z_slope;
    uint stride_x_slope = y_slope * stride_y_slope;
    uint idx_slope = min(gx, x_slope - 1) * stride_x_slope + min(gy, y_slope - 1) * stride_y_slope + min(gz, z_slope - 1);

    // Compute PReLU: f(x) = max(0, x) + slope * min(0, x)
    float data_val = in_tensor_data[idx_data];
    float slope_val = in_tensor_slope[idx_slope];
    out_tensor[gx * max(stride_x_data, stride_x_slope) + gy * max(stride_y_data, stride_y_slope) + gz] = 
        max(0.0, data_val) + slope_val * min(0.0, data_val);
}''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"PReLUOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"PReLUOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "PReLUOp requires 2 inputs: x and a"

        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

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
        assert len(input_tensors) == 2, "PReLUOp requires 2 inputs: data and slope"
        tensor_data = input_tensors[0][0]
        shape_data = input_tensors[0][1]
        tensor_slope = input_tensors[1][0]
        shape_slope = input_tensors[1][1]

        # Handle dimension alignment
        assert len(shape_slope) <= len(shape_data), "PReLUOp requires input slope to have less dimensions than input x"
        if len(shape_slope) < len(shape_data):
            new_shape_slope = [1] * (len(shape_data) - len(shape_slope)) + shape_slope
        else:
            new_shape_slope = shape_slope

        # Broadcast parameter slope if needed
        new_slope = tensor_slope
        algorithms_slope, next_tensors_slope = [], []
        if shape_data[:-2] != new_shape_slope[:-2] and not all(e == 1 for e in new_shape_slope[:-2]):
            final_shape_slope = shape_data[:-2] + list(new_shape_slope[-2:])
            new_slope = broadcast_to(tensor_slope, new_shape_slope, final_shape_slope, algorithms_slope, next_tensors_slope, self.manager)
            updated_algorithms.extend(algorithms_slope)
            new_shape_slope = final_shape_slope

        # Determine size parameters
        if len(shape_data) == 1:
            size_x_data, size_y_data, size_z_data = shape_data[0], 1, 1
            size_x_slope, size_y_slope, size_z_slope = new_shape_slope[0], 1, 1
        elif len(shape_data) == 2:
            size_x_data, size_y_data, size_z_data = shape_data[0], shape_data[1], 1
            size_x_slope, size_y_slope, size_z_slope = new_shape_slope[0], new_shape_slope[1], 1
        else:
            size_x_data = np.prod(shape_data[:-2])
            size_y_data, size_z_data = shape_data[-2], shape_data[-1]
            size_x_slope = np.prod(new_shape_slope[:-2])
            size_y_slope, size_z_slope = new_shape_slope[-2], new_shape_slope[-1]

        # Create output tensor and algorithm
        size = np.prod(shape_data)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (
            max(size_x_data, size_x_slope),
            max(size_y_data, size_y_slope),
            max(size_z_data, size_z_slope)
        )
        updated_algorithms.append(self.manager.algorithm(
            [tensor_data, new_slope, tensor_out],
            self.compiled_shader,
            workgroup,
            [size_x_data, size_y_data, size_z_data, size_x_slope, size_y_slope, size_z_slope],
            []
        ))

        return [(tensor_out, shape_data)]