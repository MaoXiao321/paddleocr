# encoding: utf-8

import os
import xml.etree.ElementTree as ET
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import random



class Process():
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def change_Suffix(self):
        """统一图片后缀"""
        file_names = os.listdir(self.data_path)
        extends = set(pd.DataFrame(file_names)[0].str.split('.', expand=True)[1]) # 拿到所有后缀
        print(f'extends:{extends}\n')
        if len(extends) > 2:
            for name in file_names:
                from_path = os.path.join(self.data_path, name)
                file_name, file_extend = os.path.splitext(from_path) # 拿到后缀
                if file_extend in ['.png', '.jpeg', '.JPG']:
                    os.rename(from_path, f'{file_name}.jpg')
            print("图片格式修改完毕！")
        print("图片格式无需修改！")
       


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", "-r", type=str, default='home/data/1')
        return parser.parse_args()
    args = parse_args()

    process = Process(args.data_path)
    process.change_Suffix()  # 改后缀
