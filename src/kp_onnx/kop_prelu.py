import kp
import numpy as np
from .shader_utils import compile_source, broadcast_to


class PReLUOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
        self.compiled_shader = compile_source('''
#version 450

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout (binding = 0) buffer buf_in_tensor_x { float in_tensor_x[]; };
layout (binding = 1) buffer buf_in_tensor_a { float in_tensor_a[]; };
layout (binding = 2) buffer buf_out_tensor { float out_tensor[]; };
layout (constant_id = 0) const float size_x_x = 0;
layout (constant_id = 1) const float size_y_x = 0;
layout (constant_id = 2) const float size_z_x = 0;
layout (constant_id = 3) const float size_x_a = 0;
layout (constant_id = 4) const float size_y_a = 0;
layout (constant_id = 5) const float size_z_a = 0;

void main()
{
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;
    uint gz = gl_GlobalInvocationID.z;

    uint x_x = uint(size_x_x);
    uint y_x = uint(size_y_x);
    uint z_x = uint(size_z_x);
    uint stride_y_x = z_x;
    uint stride_x_x = y_x * stride_y_x;
    uint idx_x = min(gx, x_x - 1) * stride_x_x + min(gy, y_x - 1) * stride_y_x + min(gz, z_x - 1);

    uint x_a = uint(size_x_a);
    uint y_a = uint(size_y_a);
    uint z_a = uint(size_z_a);
    uint stride_y_a = z_a;
    uint stride_x_a = y_a * stride_y_a;
    uint idx_a = min(gx, x_a - 1) * stride_x_a + min(gy, y_a - 1) * stride_y_a + min(gz, z_a - 1);

    // Compute PReLU: f(x) = max(0, x) + a * min(0, x)
    float x_val = in_tensor_x[idx_x];
    float a_val = in_tensor_a[idx_a];
    out_tensor[gx * max(stride_x_x, stride_x_a) + gy * max(stride_y_x, stride_y_a) + gz] = 
        max(0.0, x_val) + a_val * min(0.0, x_val);
}''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"PReLUOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"PReLUOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "PReLUOp requires 2 inputs: x and a"
        input_x, input_a = inputs[0], inputs[1]

        # Handle dimension alignment
        if input_a.ndim < input_x.ndim:
            new_shape_x = list(input_x.shape)
            new_shape_a = [1] * (input_x.ndim - input_a.ndim) + list(input_a.shape)
        else:
            new_shape_x = list(input_x.shape)
            new_shape_a = list(input_a.shape)

        # Calculate output shape based on broadcasting rules
        output_shape = [new_shape_x[i] for i in range(len(new_shape_x))]

        # Prepare input x tensor with broadcasting
        numpy_x = input_x.reshape(-1).astype(np.float32)
        tensor_x = self.manager.tensor(numpy_x)

        # Prepare parameter a tensor with broadcasting
        numpy_a = input_a.reshape(-1).astype(np.float32)
        tensor_a = self.manager.tensor(numpy_a)
        new_a = tensor_a
        algorithms_a, next_tensors_a = [], []
        if output_shape[:-2] != new_shape_a[:-2] and not all(e == 1 for e in new_shape_a[:-2]):
            final_shape_a = output_shape[:-2] + list(new_shape_a[-2:])
            new_a = broadcast_to(tensor_a, new_shape_a, final_shape_a, algorithms_a, next_tensors_a, self.manager)
            new_shape_a = final_shape_a

        # Determine size parameters for shader
        if len(new_shape_x) == 1:
            size_x_x, size_y_x, size_z_x = new_shape_x[0], 1, 1
            size_x_a, size_y_a, size_z_a = new_shape_a[0], 1, 1
        elif len(new_shape_x) == 2:
            size_x_x, size_y_x, size_z_x = new_shape_x[0], new_shape_x[1], 1
            size_x_a, size_y_a, size_z_a = new_shape_a[0], new_shape_a[1], 1
        else:
            size_x_x = np.prod(new_shape_x[:-2])
            size_y_x, size_z_x = new_shape_x[-2], new_shape_x[-1]
            size_x_a = np.prod(new_shape_a[:-2])
            size_y_a, size_z_a = new_shape_a[-2], new_shape_a[-1]

        # Create output tensor and algorithm
        size = np.prod(output_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        workgroup = (max(size_x_x, size_x_a), max(size_y_x, size_y_a), max(size_z_x, size_z_a))
        algo = self.manager.algorithm(
            [tensor_x, new_a, tensor_out],
            self.compiled_shader,
            workgroup,
            [size_x_x, size_y_x, size_z_x, size_x_a, size_y_a, size_z_a],
            []
        )

        # Execute operations
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tensor_x, tensor_a]))
        for alg in algorithms_a:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpAlgoDispatch(algo))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        # Prepare output
        outputs = [tensor_out.data().reshape(output_shape)]
        # Cleanup
        del tensor_x, tensor_a, tensor_out
        del next_tensors_a
        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert len(input_tensors) == 2, "PReLUOp requires 2 inputs: x and a"
        tensor_x = input_tensors[0][0]
        shape_x = input_tensors[0][1]
        tensor_a = input_tensors[1][0]
        shape_a = input_tensors[1][1]

        # Handle dimension alignment
        if len(shape_a) < len(shape_x):
            new_shape_x = shape_x
            new_shape_a = [1] * (len(shape_x) - len(shape_a)) + shape_a
        else:
            new_shape_x = shape_x
            new_shape_a = shape_a

        # Calculate output shape
        output_shape = [new_shape_x[i] for i in range(len(new_shape_x))]

        # Broadcast parameter a if needed
        new_a = tensor_a
        algorithms_a, next_tensors_a = [], []
        if output_shape[:-2] != new_shape_a[:-2] and not all(e == 1 for e in new_shape_a[:-2]):
            final_shape_a = output_shape[:-2] + list(new_shape_a[-2:])
            new_a = broadcast_to(tensor_a, new_shape_a, final_shape_a, algorithms_a, next_tensors_a, self.manager)
            updated_algorithms.extend(algorithms_a)
            updated_tensors.extend(next_tensors_a)
            new_shape_a = final_shape_a

        # Determine size parameters
        if len(new_shape_x) == 1:
            size_x_x, size_y_x, size_z_x = new_shape_x[0], 1, 1
            size_x_a, size_y_a, size_z_a = new_shape_a[0], 1, 1
        elif len(new_shape_x) == 2:
            size_x_x, size_y_x, size_z_x = new_shape_x[0], new_shape_x[1], 1
            size_x_a, size_y_a, size_z_a = new_shape_a[0], new_shape_a[1], 1
        else:
            size_x_x = np.prod(new_shape_x[:-2])
            size_y_x, size_z_x = new_shape_x[-2], new_shape_x[-1]
            size_x_a = np.prod(new_shape_a[:-2])
            size_y_a, size_z_a = new_shape_a[-2], new_shape_a[-1]

        # Create output tensor and algorithm
        size = np.prod(output_shape)
        tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
        updated_tensors.append(tensor_out)

        workgroup = (
            max(size_x_x, size_x_a),
            max(size_y_x, size_y_a),
            max(size_z_x, size_z_a)
        )
        updated_algorithms.append(self.manager.algorithm(
            [tensor_x, new_a, tensor_out],
            self.compiled_shader,
            workgroup,
            [size_x_x, size_y_x, size_z_x, size_x_a, size_y_a, size_z_a],
            []
        ))

        return [(tensor_out, output_shape)]