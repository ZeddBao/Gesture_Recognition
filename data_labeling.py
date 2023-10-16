# 用 mediapipe 提取图片手部关节点

import cv2
import torch
import mediapipe as mp
import numpy as np
import os
import time
import math
from PIL import Image
from tqdm import tqdm
import random

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

device = torch.device('cuda:0')

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3)

def get_hand_landmarks(img): 
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    annotated_image = img.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # 将results转为21*3的tensor
        hand_landmarks = torch.zeros((21, 3)).to(device)
        for i, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
            hand_landmarks[i, 0] = landmark.x
            hand_landmarks[i, 1] = landmark.y
            hand_landmarks[i, 2] = landmark.z
        # 将hand_landmarks改成最大最小归一化
        for i in range(3):
            hand_landmarks[:, i] = (hand_landmarks[:, i] - hand_landmarks[:, i].min()) / (hand_landmarks[:, i].max() - hand_landmarks[:, i].min())
        return annotated_image, hand_landmarks
    else:
        return img, None

def get_hand_landmarks_from_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield get_hand_landmarks(frame)
    cap.release()

def get_hand_landmarks_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield get_hand_landmarks(frame)
    cap.release()

def extract_landmarks():
    # 将data_augmented目录下的每个目录中的图片提取手部关节点
    # 然后与该目录名称转换成的one_hot编码组合成一个字典，例如该图片所在目录为'0'，则该图片的标签为tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 最后将所有图片的手部关节点和标签的字典组成一个列表，保存到data_augmented/目录下的dataset.pkl文件中
    data_path = 'data_augmented/'
    dataset = []
    for file_dir in os.listdir(data_path):
        file_path = os.path.join(data_path, file_dir) 
        # 如果file_path不是目录，则跳过
        if not os.path.isdir(file_path):
            continue
        file_list = os.listdir(file_path)
        length = len(file_list)
        for i, img_file in enumerate(file_list):
            img_file_path = os.path.join(file_path, img_file)
            img = cv2.imread(img_file_path)
            _, hand_landmarks = get_hand_landmarks(img)
            if hand_landmarks is None:
                print(f"{img_file_path} extract hand landmarks failed {i}/{length}")
                continue
            label = torch.zeros(10)
            label[int(file_dir)] = 1    # 
            dataset.append((hand_landmarks, label))
            print(f"{img_file_path} extract hand landmarks success {i}/{length}")

    torch.save(dataset, os.path.join(data_path, 'dataset_v2.pkl'))
    print('extract hand landmarks success')
    return dataset

def randomize(dataset):
    # 打乱的时候加入进度条
    dataset_randomized = []
    for i in range(len(dataset)):
        dataset_randomized.append(dataset.pop(random.randint(0, len(dataset) - 1)))
    # 保存打乱后的数据集
    torch.save(dataset_randomized, 'data_augmented/dataset_randomized_v2.pkl')
    return dataset_randomized

if __name__ == '__main__':
    dataset = extract_landmarks()
    dataset_randomized = randomize(dataset)
    print(len(dataset_randomized))