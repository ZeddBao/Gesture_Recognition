from mlp import MLP
import torch

# 1. 准备数据和模型
model = MLP(63, 128, 10)
model.load_state_dict(torch.load('ckpt/1026_01/model.pth'))
model.to('cpu')  # 量化当前只在CPU上支持
model.eval()

# 2. 定义量化配置
qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 使用FBGEMM的默认配置
model.qconfig = qconfig

# 3. 转换模型
torch.quantization.prepare(model, inplace=True)

torch.quantization.convert(model, inplace=True)

# 4. 测试量化模型
# 这里你可以运行你的推理代码，看看量化模型的性能和精度