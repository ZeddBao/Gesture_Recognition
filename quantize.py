from mlp import MLP
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools

# 1. 准备数据和模型
model = MLP(63, 128, 10)
model.load_state_dict(torch.load('ckpt/1026_01/model.pth'))
device = torch.device('cuda:0')
model.to(device)
model.eval()

# 2. 定义量化配置
model.qconfig = torch.quantization.default_qconfig

# 3. 准备模型进行量化
model = torch.quantization.prepare(model, inplace=True)

# 4. 通过模型运行代表数据，以便收集量化参数
dataset = torch.load('dataset_pkl/v2/dataset_train.pkl')
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
limited_dataloader = itertools.islice(dataloader, 32)

for i, (hand_landmarks, labels) in enumerate(tqdm(limited_dataloader)):
    current_batch_size = hand_landmarks.size(0)
    labels = labels.to(device)
    input = hand_landmarks.view(current_batch_size, -1).to(device)
    output = model(input)

# 5. 转换模型为量化模型
model = torch.quantization.convert(model, inplace=True)

# 6. 保存量化模型
torch.save(model.state_dict(), 'ckpt/1026_01/model_quantized.pth')