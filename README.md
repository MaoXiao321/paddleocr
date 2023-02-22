## 代码准备
`git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git`

## 安装依赖（paddlepaddle版本选择根据自身环境选择）
```
# 更新pip
python -m pip install --upgrade pip
# 查看当前环境的cuda版本
nvcc -V
# 参考cuda版本，选择正确的安装命令：https://www.paddlepaddle.org.cn/install/old?docurl=/documentation/docs/zh/install/pip/linux-pip.html,例如：
python -m pip install paddlepaddle-gpu==2.3.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 安装其他依赖
pip install -r requirements.txt
python setup.py install
python -m pip install paddle2onnx
python -m pip install onnxruntime==1.9.0
```

## 数据准备
对jpg文件进行标注，可以用paddleocr自带标注工具，这里使用lableme的四点标注，生成json文件。
用Process.py做一些预处理，保证数据集的后缀都是.jpg和.json。将数据集放在/home/data/1/下面。创建以下目录：
```
mkdir /home/data/json
cp /home/data/1/*.json  /home/data/json
mkdir /home/data/det_train_data /home/data/res_train_data
mkdir /home/data/det_train_data/train /home/data/det_train_data/val
mkdir /home/data/res_train_data/train /home/data/res_train_data/val
```
划分数据集，将json文件制作成paddleocr文本检测和识别的数据集格式，制作好的数据集在/home/data/下：
```
python /project/train/src_repo/PaddleOCR-release-2.6/JsonToTxt.py --data_path /home/data/1  --txt_path /home/data
rm -rf /home/data/json
```

## 训练文本检测
修改det_mv3_db.yml中的训练和验证集位置，根据需要下载预训练模型
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
修改ch_PP-OCRv3_rec.yml中的训练集、验证集和字典位置，根据需要下载预训练模型
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
