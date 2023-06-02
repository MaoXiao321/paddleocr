## 代码准备
`git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git`

## 安装依赖（paddlepaddle版本选择根据自身环境选择）
```
# 更新pip
python -m pip install --upgrade pip
# 查看当前环境的cuda版本
nvcc -V
# 参考cuda版本，选择正确的安装命令：
https://www.paddlepaddle.org.cn/install/old?docurl=/documentation/docs/zh/install/pip/linux-pip.html,
例如：python -m pip install paddlepaddle-gpu==2.3.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 安装其他依赖
pip install -r requirements.txt
python setup.py install
python -m pip install paddle2onnx
python -m pip install onnxruntime==1.9.0
```
## 数据准备
对jpg文件进行标注，可以用paddleocr自带标注工具，也可以用labelme多边形标注（生成json文件）。.jpg和.json放到/home/data/1/下面。运行以下代码准备数据集，更详细介绍见：https://blog.csdn.net/qq_39066502/article/details/130992275
```
python paddle_datasets.py
```
## 训练文本检测
修改det_mv3_db.yml中训练和验证集位置，根据需要下载预训练模型
```
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams

python tools/train.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
        Global.epoch_num=2 Global.save_epoch_step=2 Global.save_model_dir=./output/db_mv3/

python tools/export_model.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model="./output/db_mv3/latest" \
      Global.save_inference_dir="./output/det_db_inference/"

paddle2onnx --model_dir ./output/det_db_inference/ \
            --model_filename inference.pdmodel --params_filename inference.pdiparams \
            --save_file ./onnx_model/det.onnx \
            --opset_version 10 --input_shape_dict="{'x':[-1,3,-1,-1]}" --enable_onnx_checker True
```
## 训练文字识别
修改ch_PP-OCRv3_rec.yml中训练集、验证集和字典位置，根据需要下载预训练模型
```
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar
cd pretrain_models
tar -xf ch_PP-OCRv3_rec_train.tar && rm -rf ch_PP-OCRv3_rec_train.tar

python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml \
      -o Global.pretrained_model=pretrain_models/ch_PP-OCRv3_rec_train/best_accuracy \
      Global.character_dict_path=dic/brass_dict.txt Global.epoch_num=2 Global.save_epoch_step=2 Global.save_model_dir=./output/rec_ppocr_v3 \
      Train.loader.batch_size_per_card=8 Eval.loader.batch_size_per_card=8 

python tools/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o Global.pretrained_model=./output/rec_ppocr_v3/latest \
      Global.save_inference_dir=./output/rec_db_inference/ Global.character_dict_path=dic/brass_dict.txt

paddle2onnx --model_dir ./output/rec_db_inference/ \
            --model_filename inference.pdmodel --params_filename inference.pdiparams \
            --save_file ./onnx_model/rec.onnx \
            --opset_version 10 --input_shape_dict="{'x':[-1,3,-1,-1]}" --enable_onnx_checker True
```
