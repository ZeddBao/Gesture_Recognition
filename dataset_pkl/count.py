import os

import torch
from torch.utils.data import DataLoader

dataset = torch.load('dataset_pkl/v2/dataset_10000.pkl')
batch_size = 32  # 设置批量大小
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#为数据集的每个label计数，label是onehot编码
label_count = torch.zeros(10)
for i, (hand_landmarks, labels) in enumerate(dataloader):
    for label in labels:
        label_count += label
print(label_count)