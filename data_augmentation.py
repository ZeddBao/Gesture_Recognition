# 将 dataset/ 目录下每个目录中的图片分别旋转45度、90度、135度、180度、225度、270度、315度，水平翻转、垂直翻转，最后将所有图片保存到 data_augmentation/目录下

import os
import cv2
import numpy as np
from PIL import Image

def rotate(img, angle):
    height, width = img.shape[:2]
    matRotate = cv2.getRotationMatrix2D((width/2.0, height/2.0), angle, 1)
    dst = cv2.warpAffine(img, matRotate, (width, height))
    return dst

def flip(img, direction):
    if direction == 0:
        dst = cv2.flip(img, 0)  # 垂直镜像
    elif direction == 1:
        dst = cv2.flip(img, 1)  # 水平镜像
    else:
        dst = img   # 原图
    return dst

def main():
    data_path = 'dataset/'
    save_path = 'data_augmented/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        save_file_path = os.path.join(save_path, file)
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        for img_file in os.listdir(file_path):
            img_file_path = os.path.join(file_path, img_file)
            img = cv2.imread(img_file_path)
            for angle in [45, 90, 135, 180, 225, 270, 315]:
                dst = rotate(img, angle)
                save_img_file_path = os.path.join(save_file_path, str(angle) + '_' + img_file)
                cv2.imwrite(save_img_file_path, dst)
            for direction in [0, 1, 'origin']:
                dst = flip(img, direction)
                save_img_file_path = os.path.join(save_file_path, str(direction) + '_' + img_file)
                cv2.imwrite(save_img_file_path, dst)

if __name__ == '__main__':
    main()