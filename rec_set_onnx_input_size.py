import os
import onnx


file_path = 'inference/onnx/rec.onnx'
model = onnx.load(file_path)
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '1'
model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '32'
model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '150'
onnx.save(model, 'inference/onnx/rec.onnx')