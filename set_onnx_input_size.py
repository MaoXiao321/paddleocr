import os
import onnx

"""根据需要修改onnx模型的input_size"""

def set_input_size(name):
    if not os.path.exists('onnx_inference'): os.mkdir('onnx_inference') 
    file_path = f'onnx_model/{name}.onnx'
    model = onnx.load(file_path)
    if name == 'det':
        """det.onnx由（？ * 3 * ？ * ？）改为（1 * 3 * 640 * 640）"""
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '1'
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '640'
        model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '640'
        onnx.save(model, f'onnx_inference/{name}.onnx')
    if name == 'cls':
        """cls.onnx由（？ * 3 * ？ * ？）改为（1 * 3 * 48 * 192）"""
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '1'
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '48'
        model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '192'
        onnx.save(model, f'onnx_inference/{name}.onnx')
    if name == 'rec':
        """rec.onnx由（？ * 3 * ？ * ？）改为（-1 * 3 * 32 * 6623）"""
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '1'
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = '32'
        model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '6623' # 字典行数
        onnx.save(model, f'onnx_inference/{name}.onnx')

if __name__ == "__main__":
    for name in ['det','cls','rec']:
        set_input_size(name)