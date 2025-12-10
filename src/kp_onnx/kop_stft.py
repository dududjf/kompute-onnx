from __future__ import annotations
import kp
import numpy as np
from .shader_utils import compile_source


class STFTOp:
    def __init__(self, manager: kp.Manager, onesided: int = 1):
        self.manager = manager
        self.onesided = onesided
        self.compiled_shader = compile_source('''
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) buffer buf_in_x       { float in_x[]; };
layout (binding = 1) buffer buf_in_window  { float in_window[]; };
layout (binding = 2) buffer buf_out_result { float out_result[]; };

layout (constant_id = 0) const float x_len_f = 0;
layout (constant_id = 1) const float hop_length_f = 0;
layout (constant_id = 2) const float frame_length_f = 0;
layout (constant_id = 3) const float fft_length_f = 0;
layout (constant_id = 4) const float n_frames_f = 0;
layout (constant_id = 5) const float n_freqs_f = 0;
layout (constant_id = 6) const float use_window_f = 0;

#define PI 3.14159265358979323846

void main()
{
    uint fs = gl_GlobalInvocationID.x;
    uint k  = gl_GlobalInvocationID.y;
    uint batch_idx = gl_GlobalInvocationID.z;

    uint n_frames = uint(n_frames_f);
    uint n_freqs  = uint(n_freqs_f);

    uint x_len = uint(x_len_f);
    uint hop_length = uint(hop_length_f);
    uint frame_length = uint(frame_length_f);
    uint fft_length = uint(fft_length_f);
    uint use_window = uint(use_window_f);

    uint stride_freq = 2; 
    uint stride_frame = n_freqs * stride_freq;
    uint stride_batch = n_frames * stride_frame;

    uint out_idx = (batch_idx * stride_batch) + (fs * stride_frame) + (k * stride_freq);

    uint begin = (batch_idx * x_len) + (fs * hop_length);

    float sum_r = 0.0;
    float sum_i = 0.0;

    uint phase_acc = 0;

    for (uint n = 0; n < frame_length; ++n) {
        float val = 0.0;
        uint current_idx = begin + n;

        if (current_idx < (batch_idx + 1) * x_len) {
            val = in_x[current_idx];
        }

        if (use_window == 1) {
            val *= in_window[n];
        }

        uint phase_idx = phase_acc % fft_length;
        float angle = -2.0 * PI * float(phase_idx) / float(fft_length);

        sum_r += val * cos(angle);
        sum_i += val * sin(angle);

        phase_acc += k;
    }

    out_result[out_idx]     = sum_r;
    out_result[out_idx + 1] = sum_i;
}
''')

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"STFTOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"STFTOp({device_name})"

    def run(self, *inputs):
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

        tensor_x = input_tensors[0][0]
        shape_x = input_tensors[0][1]

        hop_length = int(input_tensors[1][0].data()[0])

        tensor_window = None
        shape_window = []
        if len(input_tensors) > 2 and input_tensors[2] is not None:
            tensor_window = input_tensors[2][0]
            shape_window = input_tensors[2][1]

        frame_length_input = None
        if len(input_tensors) > 3 and input_tensors[3] is not None:
            frame_length_input = int(input_tensors[3][0].data()[0])

        x_len = shape_x[-1]

        if frame_length_input is not None:
            frame_length = frame_length_input
        elif tensor_window is not None:
            frame_length = shape_window[0]
        else:
            frame_length = x_len

        fft_length = frame_length

        n_frames = 1 + (x_len - frame_length) // hop_length
        if n_frames < 0:
            n_frames = 0

        if self.onesided:
            n_freqs = fft_length // 2 + 1
        else:
            n_freqs = fft_length

        batch_size = int(np.prod(shape_x[:-1])) if len(shape_x) > 1 else 1

        output_shape = shape_x[:-1] + [n_frames, n_freqs, 2]
        total_elements = int(np.prod(output_shape))

        tensor_out = self.manager.tensor(np.zeros(total_elements, dtype=np.float32))
        updated_tensors.append(tensor_out)

        if total_elements > 0:
            use_window_val = 1.0 if tensor_window is not None else 0.0
            bind_window_tensor = tensor_window if tensor_window is not None else tensor_x

            params = [
                x_len,
                hop_length,
                frame_length,
                fft_length,
                n_frames,
                n_freqs,
                use_window_val
            ]

            workgroup = (n_frames, n_freqs, batch_size)

            updated_algorithms.append(self.manager.algorithm(
                [tensor_x, bind_window_tensor, tensor_out],
                self.compiled_shader,
                workgroup,
                params,
                []
            ))

        return [(tensor_out, output_shape)]
