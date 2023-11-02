import torch
from mlp import MLP
from tqdm import tqdm
from torch.utils.data import DataLoader

# model = MLP(63, 128, 10)
# model = MLP(63, 64, 10)
# model = MLP(63, 32, 10)
model = MLP(63, 16, 10)

# quantize
# model.qconfig = torch.quantization.default_qconfig
# model = torch.quantization.prepare(model, inplace=True)
# model = torch.quantization.convert(model, inplace=True)

# 加载模型
model.load_state_dict(torch.load('ckpt/1102_00/model.pth'))
# 载入gpu
device = torch.device('cuda:0')
# device = torch.device('cpu')
model = model.to(device)
model.eval()

def test(dataset_test):
    dataloader = DataLoader(dataset_test, batch_size=10000, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算，节省计算资源
        for i, (hand_landmarks, labels) in enumerate(tqdm(dataloader)):
            current_batch_size = hand_landmarks.size(0)
            labels = labels.to(device)
            # print('label.shape', labels.shape)
            input = hand_landmarks.view(current_batch_size, -1).to(device)
            output = model(input)
            output = torch.softmax(output, dim=1)
            correct += (output.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            total += current_batch_size
    accuracy = correct / total
    print('accuracy: {}'.format(accuracy))
    return accuracy


if __name__ == '__main__':
    dataset_test = torch.load('dataset_pkl/v2/dataset_test.pkl')
    test(dataset_test)