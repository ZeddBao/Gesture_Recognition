import os

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mlp import MLP


# 创建一个 SummaryWriter 对象
writer = SummaryWriter()

# 定义模型
model = MLP(63, 128, 10)

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
dataset = torch.load('dataset_pkl/v2/dataset_train.pkl')

# 使用 DataLoader 加载数据
batch_size = 1  # 设置批量大小
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

# 载入gpu
device = torch.device('cuda:0')
model = model.to(device)

for epoch in range(16):
    for i, (hand_landmarks, labels) in enumerate(tqdm(dataloader)):
        current_batch_size = hand_landmarks.size(0)
        labels = labels.to(device)
        # print('label.shape', labels.shape)
        input = hand_landmarks.view(current_batch_size, -1).to(device)
        # print('input.shape', input.shape)
        # 将模型输入转换成模型输出
        output = model(input)
        # 计算损失
        loss = criterion(output, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 将损失写入 TensorBoard
        writer.add_scalar('Loss/train', loss, epoch*len(dataset) + i)

    print('epoch: {}, loss: {}'.format(epoch+1, loss.item()))
    # 保存阶段性模型
    torch.save(model.state_dict(), 'ckpt/model_epoch{}.pth'.format(epoch+1))

# 保存最终模型
torch.save(model.state_dict(), 'ckpt/model.pth')
# 关闭 SummaryWriter
writer.close()
