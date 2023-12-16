import torch
from mlp import MLP
import cv2
import mediapipe as mp
from tqdm import tqdm
from torch.utils.data import DataLoader

# model = MLP(63, 128, 10)
# model = MLP(63, 64, 10)
model = MLP(63, 32, 10)
# model = MLP(63, 16, 10)
# 加载模型
model.load_state_dict(torch.load('ckpt/1101_23/model.pth'))
# 载入gpu
device = torch.device('cuda:0')
model = model.to(device)
model.eval()
model.half()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

def inference(hand_landmarks):
    with torch.no_grad():
        # input = hand_landmarks.view(1, -1).to(device)
        input = hand_landmarks.view(1, -1).to(device).half()
        output = model(input)
        # 对[1,10]的output进行softmax，得到[1,10]的output
        output = torch.softmax(output, dim=1)
    return output

def get_hand_landmarks(img):
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    annotated_image = img.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # 将results转为21*3的tensor
        hand_landmarks = torch.zeros((21, 3)).to(device)
        for i, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
            hand_landmarks[i, 0] = landmark.x
            hand_landmarks[i, 1] = landmark.y
            hand_landmarks[i, 2] = landmark.z
        # 将hand_landmarks改成最大最小归一化
        for i in range(3):
            hand_landmarks[:, i] = (hand_landmarks[:, i] - hand_landmarks[:, i].min()) / (hand_landmarks[:, i].max() - hand_landmarks[:, i].min())
        return annotated_image, hand_landmarks
    else:
        return img, None


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()  # 读取一帧图像
        # frame = cv2.flip(frame, 1)  # 左右镜像
        annotated_image, hand_landmarks = get_hand_landmarks(frame)
        if hand_landmarks is not None:
            output = inference(hand_landmarks)
            label, prob = torch.argmax(output, dim=1), torch.max(output, dim=1)[0]
            # if prob>0.9:
            print('label: {}, prob: {}'.format(label.item(), prob.item()))
        cv2.imshow('MediaPipe Hands', annotated_image)
        if cv2.waitKey(5) & 0xFF == 27: # 按Esc键退出
            break
