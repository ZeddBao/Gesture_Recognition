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


def get_hand_landmarks(img, static_image_mode=True):
    with mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=0.3) as hands:
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
                hand_landmarks[:, i] = (hand_landmarks[:, i] - hand_landmarks[:, i].min()) / (
                    hand_landmarks[:, i].max() - hand_landmarks[:, i].min())
            return annotated_image, hand_landmarks
        else:
            return img, None


def get_hand_landmarks_from_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_image, hand_landmarks = get_hand_landmarks(frame, static_image_mode=False)
        yield annotated_image, hand_landmarks
    cap.release()


def get_hand_landmarks_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield get_hand_landmarks(frame)
    cap.release()


def extract_landmarks(data_path):
    # 将data_augmented目录下的每个目录中的图片和视频提取手部关节点
    # 然后与该目录名称转换成的one_hot编码组合成一个字典，例如该图片所在目录为'0'，则该图片的标签为tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 最后将所有图片的手部关节点和标签的字典组成一个列表，保存到图片目录下的{dir}_dataset.pkl文件中
    file_list = os.listdir(data_path)
    length = len(file_list)
    dataset = []
    for i, file in enumerate(file_list):
        file_path = os.path.join(data_path, file)
        # 如果file_path不是目录，则跳过
        if os.path.isdir(file_path):
            extract_landmarks(file_path)
        elif file.endswith(('.JPG', 'jpg')):
            img = cv2.imread(file_path)
            _, hand_landmarks = get_hand_landmarks(img)
            if hand_landmarks is None:
                print(f"{file_path} extract hand landmarks failed {i+1}/{length}")
                continue
            label = torch.zeros(10)
            label[int(os.path.basename(data_path))] = 1    #
            dataset.append((hand_landmarks, label))
            print(f"{file_path} extract hand landmarks success {i+1}/{length}")
        elif file.endswith(('.mp4', 'MP4')):
            for j, (_, hand_landmarks) in enumerate(get_hand_landmarks_from_video(file_path)):
                if hand_landmarks is None:
                    continue
                label = torch.zeros(10)
                label[int(os.path.basename(data_path))] = 1
                dataset.append((hand_landmarks, label))
                print(f'---- {file_path} {j+1} frames processed')
            print(f"{file_path} extract hand landmarks success {i+1}/{length}, {j+1} frames extracted")
    if len(dataset) > 0:
        torch.save(dataset, os.path.join(data_path, f"{os.path.basename(data_path)}_dataset.pkl"))
        print(f"{data_path} extract hand landmarks Done!")

def collect_pkl(data_path):
    # 递归搜索data_path目录下的所有pkl文件，将其合并成一个大的pkl文件，返回dataset
    file_list = os.listdir(data_path)
    dataset = []
    for file in file_list:
        file_path = os.path.join(data_path, file)
        if os.path.isdir(file_path):
            dataset += collect_pkl(file_path)
        elif file.endswith('.pkl'):
            dataset += torch.load(file_path)
    return dataset

def delete_pkl(data_path):
    # 递归搜索data_path目录下的所有pkl文件，将其删除
    file_list = os.listdir(data_path)
    for file in file_list:
        file_path = os.path.join(data_path, file)
        if os.path.isdir(file_path):
            delete_pkl(file_path)
        elif file.endswith('.pkl'):
            os.remove(file_path)

def randomize(dataset):
    # 将dataset打乱，使用random.shuffle
    dataset_randomized = dataset.copy()
    random.shuffle(dataset_randomized)
    return dataset_randomized


if __name__ == '__main__':
    path = 'dataset_augmented/'

    extract_landmarks(path)
    dataset = collect_pkl(path)
    torch.save(dataset, os.path.join(path, 'dataset.pkl'))
    dataset_randomized = randomize(dataset)
    torch.save(dataset_randomized, os.path.join(path, 'dataset_randomized.pkl'))
    print('dataset length:', len(dataset_randomized))

    delete_pkl(path)