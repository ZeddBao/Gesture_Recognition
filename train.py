import torch
import torch.nn as nn
from tqdm import tqdm

from mlp import MLP


# 加入tensorboard



# 定义模型
model = MLP(63, 128, 10)

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
dataset = torch.load('data_augmented/dataset_randomized.pkl')

# 载入gpu
device = torch.device('cuda:0')
model = model.to(device)

for epoch in range(20):
    for hand_landmarks, label in tqdm(dataset):
        label = label.view(1, -1).to(device)
        input = hand_landmarks.view(1, -1).to(device)
        # 将模型输入转换成模型输出
        output = model(input)
        # 计算损失
        loss = criterion(output, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch: {}, loss: {}'.format(epoch, loss.item()))
    # 保存阶段性模型
    torch.save(model.state_dict(), 'ckpt/model_{}.pth'.format(epoch))
# 保存最终模型
torch.save(model.state_dict(), 'ckpt/model.pth')

