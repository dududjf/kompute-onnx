import numpy as np
import kp
from .shader_utils import compile_source


class LabelEncoderOp:
    def __init__(self, manager: kp.Manager, 
                 default_float=-0.0, 
                 default_int64=-1,
                 default_string='_Unused',
                 default_tensor=None,
                 keys_floats=None, 
                 keys_int64s=None,
                 keys_strings=None,
                 keys_tensor=None,
                 values_floats=None, 
                 values_int64s=None,
                 values_strings=None,
                 values_tensor=None):
        self.manager = manager
        self.default_float = default_float
        self.default_int64 = default_int64
        self.default_string = default_string
        self.default_tensor = default_tensor
        self.keys_floats = keys_floats
        self.keys_int64s = keys_int64s
        self.keys_strings = keys_strings
        self.keys_tensor = keys_tensor
        self.values_floats = values_floats
        self.values_int64s = values_int64s
        self.values_strings = values_strings
        self.values_tensor = values_tensor
        
        # Float shader
        self.compiled_shader_float = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_input   { float input_buf[];  };
layout (binding = 1) readonly  buffer buf_keys    { float keys_buf[];   };
layout (binding = 2) readonly  buffer buf_values  { float values_buf[]; };
layout (binding = 3) writeonly buffer buf_output  { float output_buf[]; };

layout (constant_id = 0) const float num_mappings_f = 0;
layout (constant_id = 1) const float default_value_f = 0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float input_val = input_buf[idx];
    int num_mappings = int(num_mappings_f);
    
    float output_val = default_value_f;
    for (int i = num_mappings - 1; i >= 0; --i) {
        if (input_val == keys_buf[i]) {
            output_val = values_buf[i];
            break;
        }
    }
    
    output_buf[idx] = output_val;
}
""")

        # Int shader
        self.compiled_shader_int = compile_source("""
#version 450
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly  buffer buf_input   { int input_buf[];  };
layout (binding = 1) readonly  buffer buf_keys    { int keys_buf[];   };
layout (binding = 2) readonly  buffer buf_values  { int values_buf[]; };
layout (binding = 3) writeonly buffer buf_output  { int output_buf[]; };

layout (constant_id = 0) const float num_mappings_f = 0;
layout (constant_id = 1) const float default_value_f = 0;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    int input_val = input_buf[idx];
    int num_mappings = int(num_mappings_f);
    int default_value = int(default_value_f);
    
    int output_val = default_value;
    for (int i = num_mappings - 1; i >= 0; --i) {
        if (input_val == keys_buf[i]) {
            output_val = values_buf[i];
            break;
        }
    }
    
    output_buf[idx] = output_val;
}
""")

    def __repr__(self):
        return f"LabelEncoderOp({self.manager.get_device_properties()['device_name']})"

    def __str__(self):
        return self.__repr__()

    def run(self, *inputs):
        # 根据属性判断输入数据类型
        use_int = self.keys_int64s is not None and len(self.keys_int64s) > 0
        dtype = np.int32 if use_int else np.float32
        
        input_tensors = []
        for inp in inputs:
            numpy_in = inp.reshape(-1).astype(dtype)
            tensor = self.manager.tensor(numpy_in)
            input_tensors.append((tensor, list(inp.shape)))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, shape_out = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([input_tensors[0][0]] + updated_tensors))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        output = tensor_out.data().reshape(shape_out)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        tensor_in, shape_in = input_tensors[0]
        
        # Determine which key-value pair to use (priority: tensor > float > int64 > string)
        use_int = False
        if self.keys_tensor is not None and self.values_tensor is not None:
            keys = self.keys_tensor.astype(np.float32)
            values = self.values_tensor.astype(np.float32)
            default_value = float(self.default_tensor) if self.default_tensor is not None else 0.0
            dtype = np.float32
        elif self.keys_floats is not None and len(self.keys_floats) > 0:
            keys = np.array(self.keys_floats, dtype=np.float32)
            values = np.array(self.values_floats, dtype=np.float32)
            default_value = self.default_float
            dtype = np.float32
        elif self.keys_int64s is not None and len(self.keys_int64s) > 0:
            keys = np.array(self.keys_int64s, dtype=np.int32)
            values = np.array(self.values_int64s, dtype=np.int32)
            default_value = self.default_int64
            dtype = np.int32
            use_int = True
        elif self.keys_strings is not None and len(self.keys_strings) > 0:
            # String type not supported in GPU shader, fallback to CPU
            assert False, "String keys are not supported in GPU implementation yet. Use CPU fallback."
        else:
            assert False, "Keys must be provided."
        
        assert len(keys) == len(values), f"Keys and values must have the same length: {len(keys)} != {len(values)}"
        
        num_mappings = len(keys)
        total_size = int(np.prod(shape_in))
        
        # Create tensors for keys and values
        tensor_keys = self.manager.tensor(keys)
        updated_tensors.append(tensor_keys)
        
        tensor_values = self.manager.tensor(values)
        updated_tensors.append(tensor_values)
        
        # Create output tensor
        tensor_out = self.manager.tensor(np.zeros(total_size, dtype=dtype))
        updated_tensors.append(tensor_out)
        
        # Select shader based on type
        compiled_shader = self.compiled_shader_int if use_int else self.compiled_shader_float
        
        # Create algorithm
        updated_algorithms.append(self.manager.algorithm(
            [tensor_in, tensor_keys, tensor_values, tensor_out],
            compiled_shader,
            (total_size, 1, 1),
            [num_mappings, default_value],
            []
        ))
        
        return [(tensor_out, shape_in)]

