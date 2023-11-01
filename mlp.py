import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层，丢弃率设为0.5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 在 fc1 和 fc2 之间使用 Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # 在 fc2 和 fc3 之间使用 Dropout
        x = self.fc3(x)
        return x
    
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(MLP, self).__init__()
#         self.quant = torch.quantization.QuantStub()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, num_classes)
#         # self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层，丢弃率设为0.5
#         self.dequant = torch.quantization.DeQuantStub()

#     def forward(self, x):
#         x = self.quant(x)
#         x = F.relu(self.fc1(x))
#         # x = self.dropout(x)  # 在 fc1 和 fc2 之间使用 Dropout
#         x = F.relu(self.fc2(x))
#         # x = self.dropout(x)  # 在 fc2 和 fc3 之间使用 Dropout
#         x = self.fc3(x)
#         x = self.dequant(x)
#         return x