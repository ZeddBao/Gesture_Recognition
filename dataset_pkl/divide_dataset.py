# 读取 dataset_pkl/v2/dataset_10000.pkl的数据集，将其分为训练集、测试集，分别保存在dataset_pkl/v2/dataset_train.pkl、dataset_pkl/v2/dataset_test.pkl中

import torch
import random

# 读取数据集
with open('dataset_pkl/v2/dataset_10000.pkl', 'rb') as f:
    dataset = torch.load(f)

# 打乱数据集
random.shuffle(dataset)

# 划分训练集、测试集
train_dataset = dataset[:90000]
test_dataset = dataset[90000:]

# 保存数据集
with open('dataset_pkl/v2/dataset_train.pkl', 'wb') as f:
    torch.save(train_dataset, f)
with open('dataset_pkl/v2/dataset_test.pkl', 'wb') as f:
    torch.save(test_dataset, f)
