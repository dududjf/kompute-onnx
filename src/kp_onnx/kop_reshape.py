import numpy as np
import kp


class ReshapeOp:
    def __init__(self, manager: kp.Manager, allowzero: int = 0):
        self.manager = manager
        self.allowzero = int(allowzero)

    def __repr__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def __str__(self):
        device_name = self.manager.get_device_properties()['device_name']
        return f"ReshapeOp({device_name})"

    def run(self, *inputs):
        assert len(inputs) >= 2, "ReshapeOp requires at least data and target_shape"
        input_tensors = []
        data_in = np.array(inputs[0], dtype=np.float32).reshape(-1)
        data_shape = list(inputs[0].shape)
        tensor_data = self.manager.tensor(data_in)
        input_tensors.append((tensor_data, data_shape))

        target_shape = np.array(inputs[1], dtype=np.int32).reshape(-1)
        tensor_shape = self.manager.tensor(target_shape)
        input_tensors.append((tensor_shape, target_shape.tolist()))

        if len(inputs) > 2:
            allowzero_val = [int(inputs[2])]
            tensor_allowzero = self.manager.tensor(np.array(allowzero_val, dtype=np.int32))
            input_tensors.append((tensor_allowzero, allowzero_val))

        updated_algorithms, updated_tensors = [], []
        output_tensor_and_shape = self.fuse(input_tensors, updated_algorithms, updated_tensors)
        tensor_out, output_shape = output_tensor_and_shape[0]

        seq = self.manager.sequence()
        seq.record(kp.OpTensorSyncDevice([t[0] for t in input_tensors]))
        for alg in updated_algorithms:
            seq.record(kp.OpAlgoDispatch(alg))
        seq.record(kp.OpTensorSyncLocal([tensor_out]))
        seq.eval()

        tensor_out.data()[:] = data_in
        output = tensor_out.data().reshape(output_shape)

        for tensor, _ in input_tensors:
            del tensor
        del updated_tensors
        return [output]

    def fuse(self, input_tensors: list[tuple[kp.Tensor, list[int]]], updated_algorithms: list[kp.Algorithm],
             updated_tensors: list[kp.Tensor]) -> list[tuple[kp.Tensor, list[int]]]:
        assert len(input_tensors) >= 2, "ReshapeOp requires data and target_shape"

        target_shape_list = input_tensors[1][1]

        if len(input_tensors) > 2:
            allowzero = int(input_tensors[2][1][0])
        else:
            allowzero = self.allowzero

        new_shape = []
        for i, dim in enumerate(target_shape_list):
            if dim == 0 and allowzero == 0:
                new_shape.append(input_tensors[0][1][i])
            else:
                new_shape.append(dim)

        neg_idx, known_prod = -1, 1
        for i, dim in enumerate(new_shape):
            if dim == -1:
                if neg_idx != -1:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_prod *= dim

        total = int(np.prod(input_tensors[0][1]))
        if neg_idx != -1:
            new_shape[neg_idx] = total // known_prod

        assert np.prod(new_shape) == total, (
            f"Reshape total elements mismatch: input={total}, new_shape={new_shape}"
        )

        tensor_out = self.manager.tensor(np.zeros(total, dtype=np.float32))
        updated_tensors.append(tensor_out)
        return [(tensor_out, new_shape)]
