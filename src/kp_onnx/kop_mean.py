import kp
import numpy as np
from .shader_utils import compile_source, broadcast_to


class MeanOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager
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
    out_tensor[gx * stride_x + gy * stride_y + gz] = (in_tensor_1[p_1] + in_tensor_2[p_2]);
}''')
        self.scale_shader = compile_source('''
#version 450
layout (local_size_x=1, local_size_y=1, local_size_z=1) in;
layout (binding=0) buffer buf_in  { float in_tensor[]; };
layout (binding=1) buffer buf_out { float out_tensor[]; };
layout (constant_id=0) const float scale = 1.0;
void main() {
    uint gx = gl_GlobalInvocationID.x;
    out_tensor[gx] = in_tensor[gx] * scale;
}
''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MeanOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"MeanOp({device_name})"

    def run(self, *inputs):
        if len(inputs) == 1:
            return [inputs[0].astype(np.float32, copy=True)]

        def _binary_add_run(x, y):
            input_1 = x
            input_2 = y
            if input_1.ndim < input_2.ndim:
                new_shape_1 = [1] * (input_2.ndim - input_1.ndim) + list(input_1.shape)
                new_shape_2 = list(input_2.shape)
            elif input_2.ndim < input_1.ndim:
                new_shape_1 = list(input_1.shape)
                new_shape_2 = [1] * (input_1.ndim - input_2.ndim) + list(input_2.shape)
            else:
                new_shape_1 = list(input_1.shape)
                new_shape_2 = list(input_2.shape)
            output_shape = []
            for i in range(len(new_shape_1)):
                if new_shape_1[i] == 1:
                    output_shape.append(new_shape_2[i])
                elif new_shape_2[i] == 1:
                    output_shape.append(new_shape_1[i])
                else:
                    assert new_shape_1[i] == new_shape_2[i], f"MeanOp requires input {i} of the same shape"
                    output_shape.append(new_shape_1[i])

            numpy_in_1 = input_1.reshape(-1).astype(np.float32, copy=False)
            tensor_in_1 = self.manager.tensor(numpy_in_1)
            new_in_1 = tensor_in_1
            algorithms_1, next_tensors_1 = [], []
            if output_shape[:-2] != new_shape_1[:-2] and not all(e == 1 for e in new_shape_1[:-2]):
                final_shape_1 = output_shape[:-2] + list(new_shape_1[-2:])
                new_in_1 = broadcast_to(tensor_in_1, new_shape_1, final_shape_1, algorithms_1, next_tensors_1, self.manager)
                new_shape_1 = final_shape_1

            numpy_in_2 = input_2.reshape(-1).astype(np.float32, copy=False)
            tensor_in_2 = self.manager.tensor(numpy_in_2)
            new_in_2 = tensor_in_2
            algorithms_2, next_tensors_2 = [], []
            if output_shape[:-2] != new_shape_2[:-2] and not all(e == 1 for e in new_shape_2[:-2]):
                final_shape_2 = output_shape[:-2] + list(new_shape_2[-2:])
                new_in_2 = broadcast_to(tensor_in_2, new_shape_2, final_shape_2,
                                        algorithms_2, next_tensors_2, self.manager)
                new_shape_2 = final_shape_2

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
                size_x_1 = np.prod(new_shape_1[:-2])
                size_y_1 = new_shape_1[-2]
                size_z_1 = new_shape_1[-1]
                size_x_2 = np.prod(new_shape_2[:-2])
                size_y_2 = new_shape_2[-2]
                size_z_2 = new_shape_2[-1]

            size = np.prod(output_shape)
            tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
            workgroup = (max(size_x_1, size_x_2), max(size_y_1, size_y_2), max(size_z_1, size_z_2))
            # “a+b”的 compiled_shader
            algo = self.manager.algorithm([new_in_1, new_in_2, tensor_out],
                                          self.compiled_shader,
                                          workgroup,
                                          [size_x_1, size_y_1, size_z_1, size_x_2, size_y_2, size_z_2],
                                          [])

            seq = self.manager.sequence()
            seq.record(kp.OpTensorSyncDevice([tensor_in_1, tensor_in_2]))
            for alg_1 in algorithms_1:
                seq.record(kp.OpAlgoDispatch(alg_1))
            for alg_2 in algorithms_2:
                seq.record(kp.OpAlgoDispatch(alg_2))
            seq.record(kp.OpAlgoDispatch(algo))
            seq.record(kp.OpTensorSyncLocal([tensor_out]))
            seq.eval()
            output = tensor_out.data()
            outputs = [output.reshape(output_shape)]

            del tensor_in_1, next_tensors_1, tensor_in_2, next_tensors_2, tensor_out
            return outputs

        # 逐个相加归约
        cur = inputs[0]
        for nxt in inputs[1:]:
            cur = _binary_add_run(cur, nxt)[0]

        # 末尾统一缩放（乘以 1/N）
        n = float(1.0 / len(inputs))

        flat = cur.reshape(-1).astype(np.float32, copy=False)
        tin = self.manager.tensor(flat)
        tout = self.manager.tensor(np.zeros_like(flat))
        algo = self.manager.algorithm([tin, tout], self.scale_shader,
                                      (flat.size, 1, 1), [n], [])
        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([tin]))
        seq.record(kp.OpAlgoDispatch(algo))
        seq.record(kp.OpTensorSyncLocal([tout]))
        seq.eval()
        out = tout.data().reshape(cur.shape)
        del tin, tout
        return [out]

    def fuse(self,
             input_tensors: list[tuple[kp.Tensor, list[int]]],
             updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        if len(input_tensors) == 1:
            return [input_tensors[0]]

        def _fuse_two_sum(lhs: tuple[kp.Tensor, list[int]],
                          rhs: tuple[kp.Tensor, list[int]]) -> tuple[kp.Tensor, list[int]]:
            input_1, shape_1 = lhs[0], list(lhs[1])
            input_2, shape_2 = rhs[0], list(rhs[1])

            # 维度对齐（补 1 维）
            if len(shape_1) < len(shape_2):
                new_shape_1 = [1] * (len(shape_2) - len(shape_1)) + shape_1
                new_shape_2 = shape_2
            elif len(shape_2) < len(shape_1):
                new_shape_1 = shape_1
                new_shape_2 = [1] * (len(shape_1) - len(shape_2)) + shape_2
            else:
                new_shape_1 = shape_1
                new_shape_2 = shape_2

            # 广播输出形状
            output_shape = []
            for i in range(len(new_shape_1)):
                if new_shape_1[i] == 1:
                    output_shape.append(new_shape_2[i])
                elif new_shape_2[i] == 1:
                    output_shape.append(new_shape_1[i])
                else:
                    assert new_shape_1[i] == new_shape_2[i], f"MeanOp requires input {i} of the same shape"
                    output_shape.append(new_shape_1[i])

            new_in_1 = input_1
            algorithms_1, next_tensors_1 = [], []
            if output_shape[:-2] != new_shape_1[:-2] and not all(e == 1 for e in new_shape_1[:-2]):
                final_shape_1 = output_shape[:-2] + list(new_shape_1[-2:])
                new_in_1 = broadcast_to(input_1, new_shape_1, final_shape_1,
                                        algorithms_1, next_tensors_1, self.manager)
                updated_algorithms.extend(algorithms_1)
                updated_tensors.extend(next_tensors_1)
                new_shape_1 = final_shape_1

            new_in_2 = input_2
            algorithms_2, next_tensors_2 = [], []
            if output_shape[:-2] != new_shape_2[:-2] and not all(e == 1 for e in new_shape_2[:-2]):
                final_shape_2 = output_shape[:-2] + list(new_shape_2[-2:])
                new_in_2 = broadcast_to(input_2, new_shape_2, final_shape_2,
                                        algorithms_2, next_tensors_2, self.manager)
                updated_algorithms.extend(algorithms_2)
                updated_tensors.extend(next_tensors_2)
                new_shape_2 = final_shape_2

            if len(new_shape_1) == 1:
                size_x_1, size_y_1, size_z_1 = new_shape_1[0], 1, 1
                size_x_2, size_y_2, size_z_2 = new_shape_2[0], 1, 1
            elif len(new_shape_1) == 2:
                size_x_1, size_y_1, size_z_1 = new_shape_1[0], new_shape_1[1], 1
                size_x_2, size_y_2, size_z_2 = new_shape_2[0], new_shape_2[1], 1
            else:
                size_x_1 = int(np.prod(new_shape_1[:-2]))
                size_y_1 = new_shape_1[-2]
                size_z_1 = new_shape_1[-1]
                size_x_2 = int(np.prod(new_shape_2[:-2]))
                size_y_2 = new_shape_2[-2]
                size_z_2 = new_shape_2[-1]

            size = np.prod(output_shape)
            tensor_out = self.manager.tensor(np.zeros(size, dtype=np.float32))
            updated_tensors.append(tensor_out)

            workgroup = (max(size_x_1, size_x_2), max(size_y_1, size_y_2), max(size_z_1, size_z_2))
            updated_algorithms.append(
                self.manager.algorithm(
                    [new_in_1, new_in_2, tensor_out],
                    self.compiled_shader,
                    workgroup,
                    [size_x_1, size_y_1, size_z_1, size_x_2, size_y_2, size_z_2],
                    []
                )
            )
            return tensor_out, output_shape

        # 所有输入两两归约
        cur = input_tensors[0]
        for i in range(1, len(input_tensors)):
            cur = _fuse_two_sum(cur, input_tensors[i])

        # 按 1/N 缩放
        tensor_out, output_shape = cur
        total_elems = np.prod(output_shape)
        out_tensor = self.manager.tensor(np.zeros(total_elems, dtype=np.float32))
        updated_tensors.append(out_tensor)

        inv_n = float(np.float32(1.0 / len(input_tensors)))

        algo_scale = self.manager.algorithm([tensor_out, out_tensor],
                                            self.scale_shader,
                                            (total_elems, 1, 1),
                                            [inv_n], [])
        updated_algorithms.append(algo_scale)

        return [(tensor_out, output_shape)]
