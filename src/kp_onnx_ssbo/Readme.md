** 这个目录保存新版本的ONNX算子实现，改动要点包括：**

1. 利用Storage Buffer Object传递算子参数，这样不需要将int32或uint32类型的参数强制转成float32后通过spec_consts或push_consts传递，避免参数进度损失；
2. 全部算子采用shader代码实现，可以利用大于1的local_size_x、local_size_y和local_size_z多GPU线程运行算子。

** 实现规范包括：**
1. 在算子实现类的fuse函数中定义param_in的kompute张量后，需要立即调用self.manager.sequence().record(kp.OpTensorSyncDevice([param_in])).eval()将参数同步到GPU显存中。
2. 仅使用X轴的算子统一定义layout (local_size_x = {LOCAL_X_1D}) in，其中LOCAL_X_1D是本目录shader_utils.py中定义的常量。
3. 使用X轴和Y轴的算子统一定义layout (local_size_x = {LOCAL_X_2D}, local_size_y = {LOCAL_Y_2D}) in，其中LOCAL_X_2D和LOCAL_Y_2D是本目录shader_utils.py中定义的常量。
4. 使用全部轴的算子统一定义layout (local_size_x = {LOCAL_X_3D}, local_size_y = {LOCAL_Y_3D}, local_size_z = {LOCAL_Z_3D}) in，其中LOCAL_X_3D、LOCAL_Y_3D和LOCAL_Z_3D是本目录shader_utils.py中定义的常量。
5. 输入参数包括算子参数统一使用layout (std430, set = 0, binding = ?) readonly buffer前缀。
6. 输出参数统一使用layout (std430, set = 0, binding = ?) writeonly buffer前缀。
7. 输入输出参数统一使用layout (std430, set = 0, binding = ?) readwrite buffer前缀。
8. 算子参数必须是layout (std430, set = 0, binding = ?)中最后一个binding。
9. 算子参数中必须包括各轴的最大坐标值，shader实现代码中必须判定当前各轴坐标不会超过对应轴的最大坐标值。
