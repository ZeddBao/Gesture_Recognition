import torch
from mlp import MLP
from torch.utils.data import DataLoader
import pytest

device = torch.device('cuda:0')
dataset_test = torch.load('dataset_pkl/v2/dataset_test.pkl')
dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True)
iterator = iter(dataloader)

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

def inference(input, model):
    return model(input)

def benchmark_model_half(benchmark, model):
    hand_landmarks, labels = next(iterator)  # get one sample from the test dataset
    input = hand_landmarks.view(1, -1).to(device).half()
    output = benchmark.pedantic(inference, args=(input, model), iterations=1, rounds=10000)
    # output = benchmark(inference, input, model)

def benchmark_model(benchmark, model):
    hand_landmarks, labels = next(iterator)  # get one sample from the test dataset
    input = hand_landmarks.view(1, -1).to(device)
    output = benchmark.pedantic(inference, args=(input, model), iterations=1, rounds=10000)
    # output = benchmark(inference, input, model)

def test_model_128(benchmark):
    benchmark_model(benchmark, model_list[0])

def test_model_64(benchmark):
    benchmark_model(benchmark, model_list[1])

def test_model_32(benchmark):
    benchmark_model(benchmark, model_list[2])

def test_model_16(benchmark):
    benchmark_model(benchmark, model_list[3])

def test_model_128_half(benchmark):
    benchmark_model_half(benchmark, model_list_half[0])

def test_model_64_half(benchmark):
    benchmark_model_half(benchmark, model_list_half[1])

def test_model_32_half(benchmark):
    benchmark_model_half(benchmark, model_list_half[2])

def test_model_16_half(benchmark):
    benchmark_model_half(benchmark, model_list_half[3])