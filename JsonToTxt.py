import json
import os
import numpy as np
from scipy.spatial.distance import cdist
import argparse
import cv2
import shutil


def order_points(pts):
    """顺时针排序box四个点"""
    pts = np.array(pts)
    sort_x = pts[np.argsort(pts[:, 0]), :]  # 按x坐标由小到大排序
    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    # Left sort
    Left = Left[np.argsort(Left[:, 1]), :]
    (lt, lb) = Left  # 拿到左上和左下
    distance = cdist(lt[np.newaxis], Right, 'euclidean')[0]  # 算左上角与右边两个点的距离
    (rb, rt) = Right[np.argsort(distance)[::-1], :]  # 距离最长的是右下
    return np.float32([lt, rt, rb, lb])


def transfrom_dict_to_str(target_dict, txt_path, flag):
    """将json转成文本检测要求的txt文件"""
    train = os.path.join(os.path.join(txt_path, 'res_train_data'), 'res_train_label.txt')
    val = os.path.join(os.path.join(txt_path, 'res_train_data'), 'res_val_label.txt')
    ftrain = open(train, 'a', encoding='utf-8')
    fval = open(val, 'a', encoding='utf-8')
    # list_target = []
    target = ''
    list_data = target_dict['shapes']
    image_path = target_dict['imagePath']
    for i, item in enumerate(list_data):
        dict1 = {}
        pts = item['points']
        if len(pts) != 4:
            continue
        points = order_points(pts)
        dict1['transcription'] = item['label']
        dict1['points'] = points.tolist()
        target = target + json.dumps(dict1,ensure_ascii=False) + ','

        # 生成文本识别需要的txt文件
        path = os.path.join(os.path.join(txt_path, 'det_train_data'))
        image = cv2.imread(os.path.join(os.path.join(path, flag), image_path))  # cv2读入后图片为BGR格式
        img_crop = get_rotate_crop_image(image, points)
        crop_name = image_path.split('.')[0] + f'_crop_{str(i)}' + '.jpg'
        result = f'{flag}/' + crop_name + '\t' + item['label'] + '\n'
        crop_path = os.path.join(os.path.join(txt_path, 'res_train_data'))
        cv2.imwrite(os.path.join(os.path.join(crop_path, flag), crop_name), img_crop)
        fval.write(result) if flag == 'val' else ftrain.write(result)
    ftrain.close()
    fval.close()
    target_str = '[' + target[:-1] + ']'
    # target_str = target_str.replace("\"", "\\"") # 将字符中的"符号提前进行转义
    # target_str = target_str.replace("'", "\"")  # 将字符中的'符号转为"
    return target_str


def get_rotate_crop_image(img, points):
    """
    将给定四个点围成的区域裁剪下来并矫正
    points = np.float32([[651.1627906976745, 2154.883720930233], [914.4186046511628, 2158.139534883721],
                         [907.9069767441861, 2214.883720930233], [647.4418604651163, 2208.837209302326]])
    """
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def split_det(data_path, txt_path):
    """划分文本检测数据集"""
    train = os.path.join(os.path.join(txt_path, 'det_train_data'), 'det_train_label.txt')
    val = os.path.join(os.path.join(txt_path, 'det_train_data'), 'det_val_label.txt')
    ftrain = open(train, 'a', encoding='utf-8')
    fval = open(val, 'a', encoding='utf-8')
    json_path = os.path.join(txt_path, 'json')
    for dirpath, dirnames, filenames in os.walk(json_path):
        for i, filename in enumerate(filenames):
            img = filename.replace('.json', '.JPG')
            with open(os.path.join(json_path, filename), encoding='utf-8') as data:
                flag = 'val' if (i + 1) % 2 == 0 else 'train'
                from_path = os.path.join(data_path, img)
                to_path = os.path.join(txt_path, 'det_train_data')
                shutil.copy(from_path, os.path.join(to_path, 'val')) if flag == 'val' \
                    else shutil.copy(from_path, os.path.join(to_path, 'train'))
                dict01 = json.load(data)
                target_str = transfrom_dict_to_str(dict01, txt_path, flag)  # 将json转成txt
                image_path = dict01['imagePath']
                result = f'{flag}/' + image_path + '\t' + target_str + '\n'
                fval.write(result) if flag == 'val' else ftrain.write(result)

    ftrain.close()
    fval.close()
    print("--------数据集制作并划分完成--------")


# def crop_img():
#     img = cv2.imread('bus.jpg')
#     y_min, y_max = 100,500
#     x_min, x_max = 100,500
#     crop = img[y_min:y_max, x_min:x_max]
#     cv2.imwrite('demo.jpg', crop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='home/data/1', type=str, help='iamges path')
    parser.add_argument('--txt_path', default='home/data', type=str, help='output txt label path')
    opt = parser.parse_args()
    split_det(opt.data_path, opt.txt_path)
