## 代码准备
`git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git`
## 安装依赖（paddlepaddle版本选择根据自身环境选择）
```
# 更新pip
python -m pip install --upgrade pip
# 查看当前环境的cuda版本
nvcc -V
# 参考cuda版本，安装正确的paddlepaddle：
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html,
例如：python -m pip install paddlepaddle==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装paddleocr
pip install "paddleocr>=2.0.1"

# 安装其他依赖
pip install -r requirements.txt
python setup.py install
python -m pip install paddle2onnx
python -m pip install onnxruntime==1.9.0
```
## 推理模型使用
```
python tools/infer/predict_det.py --image_dir doc/imgs/00111002.jpg --det_model_dir inference/ch_ppocr_server_v2.0_det_infer
python tools/infer/predict_cls.py --image_dir doc/imgs_words/ch/word_4.jpg --cls_model_dir inference/ch_ppocr_mobile_v2.0_cls_infer
python tools/infer/predict_rec.py --image_dir doc/imgs_words/ch/word_4.jpg --rec_model_dir inference/ch_ppocr_server_v2.0_rec_infer
# 也可以综合起来
python tools/infer/predict_system.py  --image_dir doc/imgs/00111002.jpg \   # doc/imgs,可处理图像集
                                      --det_model_dir inference/ch_ppocr_server_v2.0_det_infer \
                                      --rec_model_dir inference/ch_ppocr_server_v2.0_rec_infer \
                                      --cls_model_dir inference/ch_ppocr_mobile_v2.0_cls_infer \
                                      --use_angle_cls True \
                                      --use_space_char True
```                                     
## 数据准备
对jpg文件进行标注，可以用paddleocr自带标注工具，也可以用labelme多边形标注（生成json文件）。.jpg和.json放到/home/data/1/下面。运行以下代码准备数据集，更详细介绍见：https://blog.csdn.net/qq_39066502/article/details/130992275
```
python paddle_datasets.py
```
## 训练文本检测
下载预训练文件并修改对应的配置文件，主要改train和val的位置
```
python tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml \
      -o Global.pretrained_model=pretrain_models/ch_ppocr_server_v2.0_det_train/best_accuracy \
      Global.epoch_num=50 Global.save_epoch_step=20 Global.save_model_dir=output/det/ \
      Train.loader.batch_size_per_card=8 Train.loader.num_workers=2 
# 断点重开
python tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.checkpoints=output/det/latest.pdparams \
      Global.epoch_num=50 Global.save_epoch_step=20 Global.save_model_dir=output/det/ \
      Train.loader.batch_size_per_card=8 Train.loader.num_workers=2 
# 转inference模型
python tools/export_model.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml \
      -o Global.pretrained_model=output/det/best_accuracy Global.save_inference_dir=inference/det/
# 转onnx模型
paddle2onnx --model_dir inference/det/ --model_filename inference.pdmodel --params_filename inference.pdiparams \
      --save_file onnx_model/det.onnx --opset_version 10 --input_shape_dict="{'x':[-1,3,-1,-1]}" --enable_onnx_checker True
```
## 训练方向分类器
```
python tools/train.py -c configs/cls/cls_mv3.yml -o Global.pretrained_model=pretrain_models/ch_ppocr_mobile_v2.0_cls_train/best_accuracy \
        Global.epoch_num=50 Global.save_epoch_step=20 Global.save_model_dir=output/cls/ Train.loader.batch_size_per_card=8 Train.loader.num_workers=2 
python tools/export_model.py -c configs/cls/cls_mv3.yml -o Global.pretrained_model=output/cls/best_accuracy Global.save_inference_dir=inference/cls/
paddle2onnx --model_dir inference/cls/ --model_filename inference.pdmodel --params_filename inference.pdiparams \
            --save_file onnx_model/cls.onnx --opset_version 10 --input_shape_dict="{'x':[-1,3,-1,-1]}" --enable_onnx_checker True
```
## 训练文字识别
下载预训练文件并修改对应的配置文件，主要改train和val和字典的位置
```
python tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml \
      -o Global.pretrained_model=pretrain_models/ch_ppocr_server_v2.0_rec_train/best_accuracy \
      Global.character_dict_path=ppocr/utils/ppocr_keys_v1.txt Global.epoch_num=50 Global.save_epoch_step=20 Global.save_model_dir=output/rec/ \
      Train.loader.batch_size_per_card=8 Train.loader.num_workers=2 
# 转inference模型
python tools/export_model.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml -o Global.pretrained_model=output/rec/best_accuracy \
      Global.save_inference_dir=inference/rec/ Global.character_dict_path=ppocr/utils/ppocr_keys_v1.txt
paddle2onnx --model_dir inference/rec/ --model_filename inference.pdmodel --params_filename inference.pdiparams \
      --save_file onnx_model/rec.onnx --opset_version 10 --input_shape_dict="{'x':[-1,3,-1,-1]}" --enable_onnx_checker True
```
## 推理测试
```
# inference模型测试一下
python tools/infer/predict_system.py  --image_dir doc/imgs/00111002.jpg \
                                      --det_model_dir inference/det/ \
                                      --rec_model_dir inference/rec/ \
                                      --cls_model_dir inference/cls/ \
                                      --use_angle_cls True \
                                      --use_space_char True
# onnx模型测试一下    
python set_onnx_input_size.py
python tools/infer/predict_system.py --use_gpu=False --use_onnx=True \
                                    --det_model_dir=onnx_inference/det.onnx  \
                                    --rec_model_dir=onnx_inference/rec.onnx  \
                                    --cls_model_dir=onnx_inference/cls.onnx  \
                                    --image_dir=brass/20230522181835166493.jpg \
                                    --rec_char_dict_path=ppocr/utils/ppocr_keys_v1.txt                         
```