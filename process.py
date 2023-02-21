# encoding: utf-8

import os
import xml.etree.ElementTree as ET
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import random



class Process():
    def __init__(self, file_root='./data', from_ls=[], to_ls=[]):
        super().__init__()
        self.file_root = file_root
        self.from_ls = from_ls
        self.to_ls = to_ls

    def check_Suffix(self, file_root):
        """统计有多少种后缀"""
        file_names = pd.DataFrame(os.listdir(file_root))
        print(set(file_names[0].str.split('.', expand=True)[1]))
        # Suffix = [name.split('.')[-1] for name in file_names]
        # print(set(Suffix))

    def change_Suffix(self, file_root):
        """统一图片后缀"""
        file_names = os.listdir(file_root)
        for name in file_names:
            from_path = os.path.join(file_root, name)
            file_name, file_extend = os.path.splitext(from_path)
            if file_extend in ['.png', '.jpeg', '.JPG']:
                file_extend = '.jpg'
                os.rename(from_path, file_name + file_extend)
        print("图片格式修改完毕！")


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_root", "-r", type=str, default='./3/', help="")
        return parser.parse_args()
    args = parse_args()

    data_base_dir = args.image_root
    process = Process(data_base_dir)

    process.change_Suffix(data_base_dir)  # 改后缀
    process.check_Suffix(data_base_dir)  # 确认一下
