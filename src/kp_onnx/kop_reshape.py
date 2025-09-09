import kp
import numpy as np


class ReshapeOp:
    def __init__(self, manager: kp.Manager):
        self.manager = manager

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) >= 2, "ReshapeOp requires at least 2 inputs"

        input_tensors = [(inputs[0], list(inputs[0].data.shape))]

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, list(inputs[1:]),
                                            updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        output = np.array(tensor_out.data).reshape(output_shape)
        return (output,)

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]],shape_tensors: list[kp.Tensor],updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:

        tensor_in, tensor_shape = input_tensors[0]
        shape_data = shape_tensors[0].data
        new_shape = [int(x) for x in np.array(shape_data)]

        allowzero = int(np.array(shape_tensors[1].data).item()) if len(shape_tensors) > 1 else 0

        minus_one_count = new_shape.count(-1)
        if minus_one_count == 1:
            idx = new_shape.index(-1)
            new_shape[idx] = int(np.prod(tensor_shape) / np.prod([d for d in new_shape if d != -1]))
        elif minus_one_count > 1:
            raise ValueError("Only one dimension can be -1")

        if allowzero == 0:
            for idx, dim in enumerate(new_shape):
                if dim == 0:
                    new_shape[idx] = tensor_shape[idx]

        assert np.prod(new_shape) == np.prod(tensor_shape), \
            f"Cannot reshape tensor of size {np.prod(tensor_shape)} into shape {new_shape}"

        return [(tensor_in, new_shape)]
