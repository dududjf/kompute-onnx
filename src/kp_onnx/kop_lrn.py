import numpy as np
import kp
from .shader_utils import compile_source


class LRNOp:
    def __init__(self, manager: kp.Manager, alpha: float = 0.0001, beta: float = 0.75, bias: float = 1.0, size: int = 5):
        self.manager = manager
        self.alpha = alpha
        self.beta = beta
        self.bias = bias
        self.size = size
        self.compiled_shader = compile_source('''
#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(binding=0) buffer buf_in_tensor    { float in_tensor[]; };
layout(binding=1) buffer buf_out_tensor   { float out_tensor[]; };
layout(constant_id=0) const float batch_size_f   = 0;
layout(constant_id=1) const float channel_size_f = 0; 
layout(constant_id=2) const float spatial_size_f = 0;
layout(constant_id=3) const float alpha_f        = 0;
layout(constant_id=4) const float beta_f         = 0;
layout(constant_id=5) const float bias_f         = 0;
layout(constant_id=6) const float size_f         = 0;

void main() {
    uint s = gl_GlobalInvocationID.x; 
    uint c = gl_GlobalInvocationID.y; 
    uint b = gl_GlobalInvocationID.z; 
    uint minc = uint(channel_size_f); 
    uint size = uint(size_f);
    uint S    = uint(spatial_size_f);

    uint c1 = (size - 1) / 2;
    uint c2 = (size / 2) + 1;

    uint begin = 0;
    if (c >= c1) {
        begin = c - c1;
    }

    uint end = c + c2;
    if (end > minc) {
        end = minc;
    }

    uint batch_pitch = minc * S;
    uint base_offset = b * batch_pitch + s;

    float sum_sq = 0.0;
    uint read_idx = base_offset + begin * S;

    for (uint k = begin; k < end; ++k) {
        float val = in_tensor[read_idx];
        sum_sq += val * val;
        read_idx += S;
    }

    uint curr_idx = base_offset + c * S;
    float val_x = in_tensor[curr_idx];

    float denom_base = bias_f + (alpha_f / size_f) * sum_sq;
    float denom = pow(denom_base, beta_f);

    out_tensor[curr_idx] = val_x / denom;
}
''')

    def __repr__(self):
        return f"LRNOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"LRNOp({device_name})"

    def run(self, *inputs):
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(np.float32)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        if updated_algorithms:
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
        tensor_in, shape_in = input_tensors[0]
        alpha = self.alpha
        beta = self.beta
        bias = self.bias
        size = self.size

        assert len(shape_in) == 4, f"LRN only applies on 4D tensors but got shape={shape_in}"
        assert size > 0, "LRNOp.size must be > 0"

        batch_size = shape_in[0]
        channel_size = shape_in[1]
        spatial_size = np.prod(shape_in[2:])
        total_size = np.prod(shape_in)

        tensor_out = self.manager.tensor(np.zeros((total_size,), dtype=np.float32))
        updated_tensors.append(tensor_out)

        groups_x = spatial_size
        groups_y = channel_size
        groups_z = batch_size

        params = [
            batch_size,
            channel_size,
            spatial_size,
            alpha,
            beta,
            bias,
            size
        ]

        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_out],
            self.compiled_shader,
            (groups_x, groups_y, groups_z),
            params,
            []
        ))

        return [(tensor_out, shape_in)]
