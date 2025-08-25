import kp
import numpy as np
from .shader_utils import compile_source


class MatMulOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

        props = manager.get_device_properties()
        max_workgroup_invocation = props['max_work_group_invocations']
        max_workgroup_size = props['max_work_group_size']
        local_size_x = 1
        local_size_y = 1
        while 2 * local_size_x * local_size_y <= max_workgroup_invocation:
            if 2 * local_size_x <= max_workgroup_size[0]:
                local_size_x *= 2
            if 2 * local_size_y <= max_workgroup_size[1]:
                local_size_y *= 2
            elif 2 * local_size_x > max_workgroup_size[0]:  # stop if neither x nor y can be double
                break

        self.local_size_x = local_size_x
        self.local_size_y = local_size_y
        self.sizes = None
        self.workgroup = None
        self.shader = None
        self.shader_code1 = """
#version 450

layout (local_size_x = {local_size_x}, local_size_y = {local_size_y}) in;
layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};
layout (constant_id = 0) const float size_m_f = 0;
layout (constant_id = 1) const float size_k_f = 0;
layout (constant_id = 2) const float size_n_f = 0;

void main()
{{
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint size_m = uint(size_m_f);
    uint size_k = uint(size_k_f);
    uint size_n = uint(size_n_f);
    if(row >= size_m || col >= size_n) return;
    float acc = 0.0;
    uint start_1 = row * size_k;
    for(uint i = 0, start_2 = col; i < size_k; i++, start_2 += size_n)
        acc += in_tensor_1[start_1 + i] * in_tensor_2[start_2];
    out_tensor[(row * size_n) + col] = acc;
}}
"""
        self.shader_code2 = """
#version 450

layout (local_size_x = {local_size_x}, local_size_y = {local_size_y}, local_size_z = 1) in;
layout (set = 0, binding = 0) readonly buffer buf_in_tensor_1 {{ float in_tensor_1[]; }};
layout (set = 0, binding = 1) readonly buffer buf_in_tensor_2 {{ float in_tensor_2[]; }};
layout (set = 0, binding = 2) writeonly buffer buf_out_tensor {{ float out_tensor[]; }};
layout (constant_id = 0) const float size_m_f = 0;
layout (constant_id = 1) const float size_k_f = 0;
layout (constant_id = 2) const float size_n_f = 0;
layout (constant_id = 3) const float size_b_f = 0;

void main()
{{
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint batch = gl_GlobalInvocationID.z;
    uint size_m = uint(size_m_f);
    uint size_k = uint(size_k_f);
    uint size_n = uint(size_n_f);
    uint size_b = uint(size_b_f);
    if(row >= size_m || col >= size_n || batch >= size_b) return;
    float acc = 0.0;
    uint start_1 = (batch * size_m * size_k) + (row * size_k);
    uint start_2 = (batch * size_k * size_n) + col;
    for(uint i = 0; i < size_k; i++, start_2 += size_n)
        acc += in_tensor_1[start_1 + i] * in_tensor_2[start_2];
    out_tensor[(batch * size_m * size_n) + (row * size_n) + col] = acc;
}}
"""

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MatMulOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MatMulOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) == 2, "MatMulOp requires 2 inputs"
        if inputs[0].ndim >= 2 and inputs[1].ndim == 2:
            rows = np.prod(inputs[0].shape[:-1])
            cols = inputs[0].shape[-1]
            nrows = inputs[1].shape[0]
            ncols = inputs[1].shape[1]
            assert cols == nrows, f"MatMulOp requires #columns {cols} of the 1st and #rows {nrows} of the 2nd to equal"
            in_1 = inputs[0].reshape(-1).astype(np.float32)
            in_2 = inputs[1].reshape(-1).astype(np.float32)
            tensor_in_1 = self.manager.tensor(in_1)
            tensor_in_2 = self.manager.tensor(in_2)
            tensor_out = self.manager.tensor(np.zeros(rows * ncols, dtype=np.float32))

            if self.shader is None or self.sizes != [rows, cols, ncols]:
                self.sizes = [rows, cols, ncols]
                local_size_x = min(self.local_size_x, rows)
                local_size_y = min(self.local_size_y, ncols)
                # compile shader
                self.shader = compile_source(
                    self.shader_code1.format(local_size_x=local_size_x, local_size_y=local_size_y))
                self.workgroup = (
                    (rows + local_size_x - 1) // local_size_x, (ncols + local_size_y - 1) // local_size_y, 1)

            algo = self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out],
                                          self.shader, self.workgroup, self.sizes, [])
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([tensor_in_1, tensor_in_2])) \
                .record(kp.OpAlgoDispatch(algo)) \
                .record(kp.OpTensorSyncLocal([tensor_out])) \
                .eval()
            out_shape = inputs[0].shape[:-1] + (ncols,)
            outputs = [tensor_out.data().reshape(out_shape)]
            del tensor_in_1
            del tensor_in_2
            del tensor_out

        else:
            assert 2 < inputs[0].ndim == inputs[1].ndim and inputs[0].shape[:-2] == inputs[1].shape[:-2], \
                f"MatMulOp requires the prefix dimensions {inputs[0].shape[:-2]} and {inputs[1].shape[:-2]} to equal"
            rows = inputs[0].shape[-2]
            cols = inputs[0].shape[-1]
            nrows = inputs[1].shape[-2]
            ncols = inputs[1].shape[-1]
            assert cols == nrows, f"MatMulOp requires #columns {cols} of the 1st and #rows {nrows} of the 2nd to equal"
            blocks = np.prod(inputs[0].shape[:-2])
            in_1 = inputs[0].reshape(-1).astype(np.float32)
            in_2 = inputs[1].reshape(-1).astype(np.float32)
            tensor_in_1 = self.manager.tensor(in_1)
            tensor_in_2 = self.manager.tensor(in_2)
            tensor_out = self.manager.tensor(np.zeros(blocks * rows * ncols, dtype=np.float32))

            if self.shader is None or self.sizes != [rows, cols, ncols, blocks]:
                self.sizes = [rows, cols, ncols, blocks]
                local_size_x = min(self.local_size_x, rows)
                local_size_y = min(self.local_size_y, ncols)
                # compile shader
                self.shader = compile_source(
                    self.shader_code2.format(local_size_x=local_size_x, local_size_y=local_size_y))
                self.workgroup = (
                    (rows + local_size_x - 1) // local_size_x, (ncols + local_size_y - 1) // local_size_y, blocks)

            algo = self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out],
                                          self.shader, self.workgroup, self.sizes, [])
            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([tensor_in_1, tensor_in_2])) \
                .record(kp.OpAlgoDispatch(algo)) \
                .record(kp.OpTensorSyncLocal([tensor_out])) \
                .eval()
            out_shape = inputs[0].shape[:-1] + (ncols,)
            outputs = [tensor_out.data().reshape(out_shape)]
            del tensor_in_1
            del tensor_in_2
            del tensor_out

        return outputs

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert len(input_tensors) == 2, "MatMulOp requires 2 inputs"
        tensor_in_1 = input_tensors[0][0]
        tensor_in_2 = input_tensors[1][0]
        shape_1 = input_tensors[0][1]
        shape_2 = input_tensors[1][1]

        if len(shape_1) >= 2 and len(shape_2) == 2:
            rows = np.prod(shape_1[:-1])
            cols = shape_1[-1]
            nrows = shape_2[0]
            ncols = shape_2[1]
            assert cols == nrows, f"MatMulOp requires #columns {cols} of the 1st and #rows {nrows} of the 2nd to equal"
            tensor_out = self.manager.tensor(np.zeros(rows * ncols, dtype=np.float32))

            if self.shader is None or self.sizes != [rows, cols, ncols]:
                self.sizes = [rows, cols, ncols]
                local_size_x = min(self.local_size_x, rows)
                local_size_y = min(self.local_size_y, ncols)
                # compile shader
                self.shader = compile_source(
                    self.shader_code1.format(local_size_x=local_size_x, local_size_y=local_size_y))
                self.workgroup = (
                    (rows + local_size_x - 1) // local_size_x, (ncols + local_size_y - 1) // local_size_y, 1)

            updated_tensors.append(tensor_out)
            updated_algorithms.append(self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out],
                                                             self.shader, self.workgroup, self.sizes, []))

            output_shape = shape_1[:-1] + [ncols]
            return [(tensor_out, output_shape)]

        else:
            assert 2 < len(shape_1) == len(shape_2) and shape_1[:-2] == shape_2[:-2], \
                f"MatMulOp requires the prefix dimensions {shape_1[:-2]} and {shape_2[:-2]} to equal"
            rows = shape_1[-2]
            cols = shape_1[-1]
            nrows = shape_2[-2]
            ncols = shape_2[-1]
            assert cols == nrows, f"MatMulOp requires #columns {cols} of the 1st and #rows {nrows} of the 2nd to equal"
            blocks = np.prod(shape_1[:-2])
            tensor_out = self.manager.tensor(np.zeros(blocks * rows * ncols, dtype=np.float32))

            if self.shader is None or self.sizes != [rows, cols, ncols, blocks]:
                self.sizes = [rows, cols, ncols, blocks]
                local_size_x = min(self.local_size_x, rows)
                local_size_y = min(self.local_size_y, ncols)
                # compile shader
                self.shader = compile_source(
                    self.shader_code2.format(local_size_x=local_size_x, local_size_y=local_size_y))
                self.workgroup = (
                    (rows + local_size_x - 1) // local_size_x, (ncols + local_size_y - 1) // local_size_y, blocks)

            updated_tensors.append(tensor_out)
            updated_algorithms.append(self.manager.algorithm([tensor_in_1, tensor_in_2, tensor_out],
                                                             self.shader, self.workgroup, self.sizes, []))

            output_shape = shape_1[:-1] + [ncols]
            return [(tensor_out, output_shape)]
