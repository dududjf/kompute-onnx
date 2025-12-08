import numpy as np
import kp
from .shader_utils import compile_source


class UniqueOp:
    def __init__(self, manager: kp.Manager, axis=None, sorted=1):
        self.manager = manager
        self.axis = axis
        self.sorted = sorted  # Only sorted=1 is supported
        
        # Shader for Bitonic Sort (sort indices by lexicographic order)
        self.compiled_shader_sort_indices = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer buf_input { float tensor_in[]; };
layout (binding = 1) buffer buf_idx_sorted { uint tensor_index_sorted[]; };

layout (constant_id = 0) const float size_f = 0;
layout (constant_id = 1) const float leading_f = 0;
layout (constant_id = 2) const float trailing_f = 0;
layout (constant_id = 3) const float k_f = 0;
layout (constant_id = 4) const float j_f = 0;

bool lex_less(uint su, uint sv, uint size, uint leading, uint trailing)
{
    for (uint ld = 0, idx_u = su * trailing, idx_v = sv * trailing;
         ld < leading;
         ++ld, idx_u += size * trailing, idx_v += size * trailing) {
        uint off_u = idx_u;
        uint off_v = idx_v;
        for (uint tt = 0; tt < trailing; ++tt, off_u++, off_v++) {
            float a = tensor_in[off_u];
            float b = tensor_in[off_v];
            if (a < b) return true;
            if (a > b) return false;
        }
    }
    return false;
}

void main()
{
    uint tid = gl_GlobalInvocationID.x;
    uint size = uint(size_f);
    uint leading = uint(leading_f);
    uint trailing = uint(trailing_f);
    uint k = uint(k_f);
    uint j = uint(j_f);
    
    if (tid >= size) return;
    uint ixj = tid ^ j;
    
    if (ixj > tid && ixj < size) {
        uint u = tensor_index_sorted[tid];
        uint v = tensor_index_sorted[ixj];
        bool should_swap = lex_less(v, u, size, leading, trailing);
        
        if ((tid & k) == 0) {
            if (should_swap) {
                tensor_index_sorted[tid] = v;
                tensor_index_sorted[ixj] = u;
            }
        } else {
            if (!should_swap) {
                tensor_index_sorted[tid] = v;
                tensor_index_sorted[ixj] = u;
            }
        }
    }
}
""")

        # 在排序后的索引数组中，找出所有唯一的 slice，统计每个 slice 的出现次数和首次出现的最小索引。
        self.compiled_shader_detect_runs = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Inputs (readonly)
layout (binding = 0) readonly buffer buf_input { float input_buf[]; };
layout (binding = 1) readonly buffer buf_index_sorted { uint index_sorted[]; };

// Outputs (writeonly or buffer for read-modify-write)
layout (binding = 2) buffer buf_indices_out { uint indices_out[]; };
layout (binding = 3) buffer buf_counts_out { uint counts_out[]; };
layout (binding = 4) writeonly buffer buf_num_runs { uint num_runs_buf[]; };

layout (constant_id = 0) const float size_f = 0;
layout (constant_id = 1) const float leading_f = 0;
layout (constant_id = 2) const float trailing_f = 0;

bool slices_equal(uint su, uint sv, uint block, uint leading, uint trailing)
{
    for (uint ld = 0, idx_u = su * trailing, idx_v = sv * trailing;
         ld < leading;
         ++ld, idx_u += block, idx_v += block) {
        uint off_u = idx_u;
        uint off_v = idx_v;
        for (uint tt = 0; tt < trailing; ++tt, off_u++, off_v++) {
            float a = input_buf[off_u];
            float b = input_buf[off_v];
            if (a != b) return false;
        }
    }
    return true;
}

void main()
{
    uint size = uint(size_f);
    uint leading = uint(leading_f);
    uint trailing = uint(trailing_f);

    uint num_runs = 0;
    for (uint u = 0; u < size; ++u) {
        uint cur_idx = index_sorted[u];
        bool is_head = (u == 0) ? true : !slices_equal(cur_idx, index_sorted[u - 1], size * trailing, leading, trailing);
        
        if (is_head) {
            indices_out[num_runs] = cur_idx;
            counts_out[num_runs] = 1;
            num_runs++;
        } else {
            counts_out[num_runs - 1] += 1;
            if (cur_idx < indices_out[num_runs - 1]) {
                indices_out[num_runs - 1] = cur_idx;
            }
        }
    }
    
    num_runs_buf[0] = num_runs;
}
""")

        # Shader for Stage 1: Copy unique values in sorted order
        self.compiled_shader_copy_unique = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Inputs (readonly)
layout (binding = 0) readonly buffer buf_input { float input_buf[]; };
layout (binding = 1) readonly buffer buf_index_sorted { uint index_sorted[]; };
layout (binding = 2) readonly buffer buf_counts_out { uint counts_out[]; };
layout (binding = 3) readonly buffer buf_num_runs { uint num_runs_buf[]; };

// Outputs (writeonly)
layout (binding = 4) writeonly buffer buf_unique_out { float unique_out[]; };
layout (binding = 5) writeonly buffer buf_inverse_out { uint inverse_out[]; };

layout (constant_id = 0) const float size_f = 0;
layout (constant_id = 1) const float leading_f = 0;
layout (constant_id = 2) const float trailing_f = 0;

void copy_slice_tr(uint s_from, uint s_to, uint tr, uint size, uint leading, uint trailing)
{
    for (uint ld = 0, src = s_from * trailing + tr, dst = s_to * trailing + tr;
         ld < leading;
         ++ld, src += size * trailing, dst += size * trailing) {
        unique_out[dst] = input_buf[src];
    }
}

void main()
{
    uint tr = gl_GlobalInvocationID.y;
    
    uint size = uint(size_f);
    uint leading = uint(leading_f);
    uint trailing = uint(trailing_f);
    
    uint num_runs = num_runs_buf[0];
    
    uint start = 0;
    for (uint r = 0; r < num_runs; ++r) {
        uint run_len = counts_out[r];
        
        copy_slice_tr(index_sorted[start], r, tr, size, leading, trailing);
        
        if (tr == 0) {
            for (uint k = 0; k < run_len; ++k) {
                uint orig = index_sorted[start + k];
                inverse_out[orig] = r;
            }
        }
        start += run_len;
    }
}
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

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([t[0] for t in output_tensor_and_shape]))
        seq.eval()

        output_list = []
        for tensor, output_shape in output_tensor_and_shape:
            output = tensor.data().reshape(output_shape)
            output_list.append(output)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return output_list

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert self.sorted == 1, "Only sorted=1 is supported in this UniqueOp implementation."
        tensor_in, shape_in = input_tensors[0]

        if self.axis is not None:
            axis = self.axis + len(shape_in) if self.axis < 0 else self.axis
            size = shape_in[axis]
            leading = int(np.prod(shape_in[:axis])) if axis > 0 else 1
            trailing = int(np.prod(shape_in[axis+1:])) if axis + 1 < len(shape_in) else 1
        else:
            # axis=None: flatten the array, treat as 1D
            size = int(np.prod(shape_in))
            leading = 1
            trailing = 1
            shape_in = [size]  # Override shape for 1D processing

        # Step 1: Sort indices using Bitonic Sort O(n log^2 n)
        tensor_index_sorted = self.manager.tensor_t(np.arange(size, dtype=np.uint32))
        updated_tensors.append(tensor_index_sorted)
        
        # Find next power of 2 >= size for proper bitonic sort
        n_pow2 = 1
        while n_pow2 < size:
            n_pow2 *= 2
        
        k = 2
        while k <= n_pow2:
            j = k >> 1
            while j > 0:
                updated_algorithms.append(self.manager.algorithm(
                    [tensor_in, tensor_index_sorted],
                    self.compiled_shader_sort_indices,
                    (n_pow2, 1, 1),
                    [size, leading, trailing, k, j],
                    []
                ))
                j >>= 1
            k <<= 1

        tensor_unique_out = self.manager.tensor(np.zeros(int(np.prod(shape_in)), dtype=np.float32))
        tensor_indices_out = self.manager.tensor_t(np.zeros(size, dtype=np.uint32))
        tensor_counts_out = self.manager.tensor_t(np.zeros(size, dtype=np.uint32))
        tensor_inverse_out = self.manager.tensor_t(np.zeros(size, dtype=np.uint32))
        tensor_num_runs = self.manager.tensor_t(np.zeros(1, dtype=np.uint32))
        updated_tensors.extend([tensor_unique_out, tensor_indices_out, tensor_counts_out, tensor_inverse_out, tensor_num_runs])
        
        # Step 2: Detect runs (single-threaded)
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_index_sorted, tensor_indices_out, tensor_counts_out, tensor_num_runs],
            self.compiled_shader_detect_runs,
            (1, 1, 1),
            [size, leading, trailing],
            []
        ))
        
        # Step 3: Copy unique values (parallel across trailing dimension)
        # Tensor order: inputs first, then outputs
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_index_sorted, tensor_counts_out, tensor_num_runs,
             tensor_unique_out, tensor_inverse_out],
            self.compiled_shader_copy_unique,
            (1, trailing, 1),
            [size, leading, trailing],
            []
        ))

        return [(tensor_unique_out, shape_in), (tensor_indices_out, [size]),
                (tensor_inverse_out, [size]), (tensor_counts_out, [size])]
