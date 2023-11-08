import torch
from mlp import MLP, MLP2
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device('cuda:0')

model_list_half = [MLP(63, 128, 10), MLP(63, 64, 10), MLP(63, 32, 10), MLP(63, 16, 10)]
model_list_half[0].load_state_dict(torch.load('ckpt/1027_02/model.pth'))
model_list_half[1].load_state_dict(torch.load('ckpt/1101_22/model.pth'))
model_list_half[2].load_state_dict(torch.load('ckpt/1101_23/model.pth'))
model_list_half[3].load_state_dict(torch.load('ckpt/1102_00/model.pth'))
for model in model_list_half:
    model = model.to(device)
    model.eval()
    model.half()

model_list = [MLP(63, 128, 10), MLP(63, 64, 10), MLP(63, 32, 10), MLP(63, 16, 10)]
model_list[0].load_state_dict(torch.load('ckpt/1027_02/model.pth'))
model_list[1].load_state_dict(torch.load('ckpt/1101_22/model.pth'))
model_list[2].load_state_dict(torch.load('ckpt/1101_23/model.pth'))
model_list[3].load_state_dict(torch.load('ckpt/1102_00/model.pth'))
for model in model_list:
    model = model.to(device)
    model.eval()

model2 = MLP2(63, 16, 10)
model2.load_state_dict(torch.load('ckpt/1108_19/model.pth'))
model2 = model2.to(device)
model2.eval()

# quantize
# model.qconfig = torch.quantization.default_qconfig
# model = torch.quantization.prepare(model, inplace=True)
# model = torch.quantization.convert(model, inplace=True)

def test(dataset_test, model, half):
    dataloader = DataLoader(dataset_test, batch_size=10000, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算，节省计算资源
        for i, (hand_landmarks, labels) in enumerate(tqdm(dataloader)):
            current_batch_size = hand_landmarks.size(0)
            labels = labels.to(device)
            # print('label.shape', labels.shape)
            if half:
                input = hand_landmarks.view(current_batch_size, -1).to(device).half()
            else:
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
    for model in model_list:
        test(dataset_test, model, False)
    for model in model_list_half:
        test(dataset_test, model, True)

    test(dataset_test, model2, False)