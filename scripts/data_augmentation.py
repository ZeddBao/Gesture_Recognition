# 将 dataset/ 目录下每个目录中的图片分别旋转45度、90度、135度、180度、225度、270度、315度，水平翻转、垂直翻转，最后将所有图片保存到 data_augmentation/目录下

import os
import torch
import cv2
from wand.image import Image as wi
import numpy as np
from PIL import Image
from tqdm import tqdm


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
    data_path = 'dataset_origin/'
    save_path = 'dataset_augmented_buffer/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        save_file_path = os.path.join(save_path, file)
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        for img_file in tqdm(os.listdir(file_path)):
            img_file_path = os.path.join(file_path, img_file)
            if img_file.endswith('.heic'):
                with wi(filename=img_file_path) as image:
                    image.format = 'jpeg'
                    img_data = np.asarray(
                        bytearray(image.make_blob()), dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
            elif img_file.endswith(('.jpg', '.JPG')):
                img = cv2.imread(img_file_path)
            else:
                continue
            for angle in [45, 90, 135, 180, 225, 270, 315]:
                dst = rotate(img, angle)
                save_img_file_path = os.path.join(save_file_path, str(
                    angle) + '_' + img_file.replace('.heic', '.jpg'))
                cv2.imwrite(save_img_file_path, dst)
            for direction in [0, 1, 'origin']:
                dst = flip(img, direction)
                save_img_file_path = os.path.join(save_file_path, str(
                    direction) + '_' + img_file.replace('.heic', '.jpg'))
                cv2.imwrite(save_img_file_path, dst)


def augment(dataset, class_num):
    # 将输入的数据和标签分别存储在两个大的 Tensor 中
    data_tensor = torch.stack([item[0] for item in dataset])
    label_tensor = torch.stack([item[1] for item in dataset])

    # 创建一个列表，用于存储增强后的数据和标签
    augmented_data = []
    augmented_labels = []

    for i in range(10):
        # 获取当前类别的数据和标签
        current_data = data_tensor[label_tensor.argmax(dim=1) == i]
        current_labels = label_tensor[label_tensor.argmax(dim=1) == i]
        current_num = current_data.shape[0]

        # 如果当前类别的数量已经达到 class_num，直接添加到增强数据列表
        if current_num >= class_num:
            augmented_data.append(current_data[:class_num].cpu())
            augmented_labels.append(current_labels[:class_num].cpu())
        else:
            # 否则，需要生成额外的数据
            augmented_data.append(current_data.cpu())
            augmented_labels.append(current_labels.cpu())

            for _ in tqdm(range(class_num - current_num)):
                # 随机选择一个数据进行增强
                idx = torch.randint(0, current_num, (1,))
                hand_landmarks = current_data[idx].reshape(21, 3).cpu().numpy()

                # 生成一个随机的 3x3 正交矩阵
                random_matrix = np.random.rand(3, 3)
                Q, _ = np.linalg.qr(random_matrix)

                # 使用生成的正交矩阵对 hand_landmarks 进行变换
                hand_landmarks = hand_landmarks.dot(Q)
                # 将hand_landmarks改成最大最小归一化
                for i in range(3):
                    hand_landmarks[:, i] = (hand_landmarks[:, i] - hand_landmarks[:, i].min()) / (
                        hand_landmarks[:, i].max() - hand_landmarks[:, i].min())

                hand_landmarks = torch.from_numpy(
                    hand_landmarks).reshape(1, 21, 3)

                # 添加增强的数据和标签到列表
                augmented_data.append(hand_landmarks.to(torch.float32))
                augmented_labels.append(current_labels[idx])

    # 将列表转换为 Tensor
    augmented_data = torch.cat(augmented_data, dim=0)
    augmented_labels = torch.cat(augmented_labels, dim=0)

    return list(zip(augmented_data, augmented_labels))


if __name__ == '__main__':
    # main()
    dataset = torch.load('dataset_pkl/v2/dataset.pkl')
    dataset_augmented = augment(dataset, 10000)
    print(sum([item[1] for item in dataset_augmented]))
    print(dataset_augmented[0][0].shape, dataset_augmented[0][1].shape)
    torch.save(dataset_augmented, 'dataset_pkl/v2/dataset_10000.pkl')
