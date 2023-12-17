from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QCoreApplication
import sys
import cv2
import mediapipe as mp
import math
import random
from decimal import Decimal
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QMessageBox
import torch
import torch.nn as nn
import torch.nn.functional as F
# 全局变量用于存储手势识别结果和生成的数字
PRINT_LOG = True
gesture_result = None
generated_number = None
generated_color = None
generated_time = Decimal("0.0")
true_cnt = Decimal("0.0")
total_cnt = 0
add_flag = 0
compare_flag = 0
flash_flag = 1
level = 0
model_flag = 0



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        # self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层，丢弃率设为0.5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)  # 在 fc1 和 fc2 之间使用 Dropout
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)  # 在 fc2 和 fc3 之间使用 Dropout
        x = self.fc3(x)
        return x


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model_2 = MLP(63, 32, 10)
# 加载模型
model_2.load_state_dict(torch.load('ckpt/1101_23/model.pth'))
# 载入gpu
device = torch.device('cuda:0')
model_2 = model_2.to(device)
model_2.eval()
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

class VideoThread(QThread):
    frame_ready = pyqtSignal(object)
    cap = cv2.VideoCapture(0)

    def __init__(self):
        super().__init__()
        init_landmarks = torch.zeros((21, 3)).to(device)
        init_output = self.inference(init_landmarks)


    def vector_2d_angle(self, v1, v2):
        '''
            求解二维向量的角度
        '''
        v1_x = v1[0]
        v1_y = v1[1]
        v2_x = v2[0]
        v2_y = v2[1]
        try:
            angle_ = math.degrees(math.acos(
                (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
        except:
            angle_ = 65535.
        if angle_ > 180.:
            angle_ = 65535.
        return angle_

    def hand_angle(self, hand_):
        '''
            获取对应手相关向量的二维角度,根据角度确定手势
        '''
        angle_list = []
        # ---------------------------- thumb 大拇指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
            ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
        )
        angle_list.append(angle_)
        # ---------------------------- index 食指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
            ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
        )
        angle_list.append(angle_)
        # ---------------------------- middle 中指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
            ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
        )
        angle_list.append(angle_)
        # ---------------------------- ring 无名指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
            ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
        )
        angle_list.append(angle_)
        # ---------------------------- pink 小拇指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
            ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
        )
        angle_list.append(angle_)
        return angle_list

    def h_gesture(self, angle_list):
        global gesture_result
        '''
            # 二维约束的方法定义手势
            # fist five gun love one six three thumbup yeah
        '''
        thr_angle = 65.
        thr_angle_thumb = 53.
        thr_angle_s = 49.
        gesture_str = None
        '''
        print("大拇指角度： {}\n 食指角度 ： {}\n中指角度 ： {}\n 无名指角度 ： {}\n小拇指角度 ： {}".format(angle_list[0],
                                                                                                        angle_list[1],
                                                                                                        angle_list[2],
                                                                                                        angle_list[3],
                                                                                                        angle_list[4]))
        '''
        if 65535. not in angle_list:
            if (angle_list[0] > 65.) and (angle_list[1] > 65.) and (angle_list[2] > 150.) \
                    and (angle_list[3] > 150.) and (angle_list[4] > 150.):
                gesture_str = "0"
                gesture_result = 0
            elif (angle_list[0] > 5) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) \
                    and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "1"
                gesture_result = 1
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) \
                    and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "2"
                gesture_result = 2
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) \
                    and (angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
                gesture_str = "3"
                gesture_result = 3
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) \
                    and (angle_list[3] < thr_angle) and (angle_list[4] < thr_angle):
                gesture_str = "4"
                gesture_result = 4
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) \
                    and (angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
                gesture_str = "5"
                gesture_result = 5
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) \
                    and (angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
                gesture_str = "6"
                gesture_result = 6
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) \
                    and (angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
                gesture_str = "7"
                gesture_result = 7
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) \
                    and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "8"
                gesture_result = 8
            elif (angle_list[0] > thr_angle_s) and (angle_list[1] < 90.) and (angle_list[2] > thr_angle) \
                    and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "9"
                gesture_result = 9
        return gesture_str

    def inference(self, hand_landmarks):
        with torch.no_grad():
            input = hand_landmarks.view(1, -1).to(device)
            output = model_2(input)
            # 对[1,10]的output进行softmax，得到[1,10]的output
            output = torch.softmax(output, dim=1)
        return output
    def get_hand_landmarks(self, img):
        results = hands.process(img)
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
                hand_landmarks[:, i] = (hand_landmarks[:, i] - hand_landmarks[:, i].min()) / (
                            hand_landmarks[:, i].max() - hand_landmarks[:, i].min())
            return annotated_image, hand_landmarks
        else:
            return img, None


    def run(self):
        global flash_flag, level, generated_number, gesture_result, generated_color, total_cnt, true_cnt, false_cnt, add_flag, compare_flag

        while True:
            start = cv2.getTickCount()
            ret, frame = self.cap.read()             # 读取一帧图像
            if not ret:
                print("Failed to grab frame")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
            frame = cv2.flip(frame, 1)          # 左右镜像
            if model_flag == 1: #模型1
                results = hands.process(frame)    # 将图像传入模型检测
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        hand_local = []
                        for i in range(21):
                            x = hand_landmarks.landmark[i].x * frame.shape[1]
                            y = hand_landmarks.landmark[i].y * frame.shape[0]
                            hand_local.append((x, y))
                        if hand_local:
                            angle_list = self.hand_angle(hand_local)
                            gesture_str = self.h_gesture(angle_list)
                            cv2.putText(frame, gesture_str, (0, 100), 0, 1.3, (0, 0, 255), 3)
                    compare_flag = 1
            elif model_flag == 2:#模型2 训练的模型
                frame, hand_landmarks = self.get_hand_landmarks(frame)
                if hand_landmarks is not None:
                    output = self.inference(hand_landmarks)
                    label, prob = torch.argmax(output, dim=1), torch.max(output, dim=1)[0]
                    if prob > 0.9:
                        gesture_result = label.item()
                        cv2.putText(frame, str(label.item()) + ", Prob: {:.2f}".format(prob[0]), (0, 100), 0, 1.3, (0, 0, 255), 3)
                    else:
                        gesture_result = None
                    # print(label.item())
                    # print('label: {}, prob: {}'.format(label.item(), prob.item()))
                    compare_flag = 1
                # results = hands.process(frame)
                # if results.multi_hand_landmarks:
                #     for hand_landmarks in results.multi_hand_landmarks:
                #         mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                #     # 将results转为21*3的tensor
                #     hand_landmarks = torch.zeros((21, 3)).to(device)
                #     for i, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                #         hand_landmarks[i, 0] = landmark.x
                #         hand_landmarks[i, 1] = landmark.y
                #         hand_landmarks[i, 2] = landmark.z
                #     # 将hand_landmarks改成最大最小归一化
                #     for i in range(3):
                #         hand_landmarks[:, i] = (hand_landmarks[:, i] - hand_landmarks[:, i].min())/(hand_landmarks[:, i].max() - hand_landmarks[:, i].min())
                #     with torch.no_grad():
                #         input = hand_landmarks.view(1, -1).to(device)
                #         output = model_2(input)
                #         output = torch.softmax(output, dim=1)
                #     label, prob = torch.argmax(output, dim=1), torch.max(output, dim=1)[0]
                #     cv2.putText(frame, label.item(), (0, 100), 0, 1.3, (0, 0, 255), 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB转BGR
            if level == 0:
                cv2.putText(frame, str(generated_number), (280, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
            elif level == 1:
                cv2.putText(frame, "miss", (280, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
            elif level == 2:
                cv2.putText(frame, "good", (280, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
            elif level == 3:
                cv2.putText(frame, "perfect", (280, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
            end = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end - start)
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
            self.frame_ready.emit(frame)
    
    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.quit()
        self.wait()



class Ui_Form(object):
    started = pyqtSignal(object)

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(816, 486)
        Form.setStyleSheet("background-color: rgb(0, 85, 255);")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.image = QtWidgets.QLabel(Form)
        self.image.setMinimumSize(QtCore.QSize(640, 400))
        self.image.setStyleSheet("background-color: rgb(85, 170, 255);\n"
                                 "font: 16pt \"Arial\";\n"
                                 "color:rgb(255, 255, 255)")
        self.image.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.image.setObjectName("image")
        self.horizontalLayout_6.addWidget(self.image)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.model_check = QtWidgets.QCheckBox(Form)
        self.model_check.setObjectName("model_check")
        self.horizontalLayout_5.addWidget(self.model_check)
        self.model_check_2 = QtWidgets.QCheckBox(Form)
        self.model_check_2.setObjectName("model_check_2")
        self.horizontalLayout_5.addWidget(self.model_check_2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        spacerItem = QtWidgets.QSpacerItem(20, 100, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_4.addItem(spacerItem)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(30)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.timeout = QtWidgets.QLabel(Form)
        self.timeout.setMinimumSize(QtCore.QSize(60, 60))
        font = QtGui.QFont()
        font.setFamily("华文中宋")
        font.setPointSize(28)
        self.timeout.setFont(font)
        self.timeout.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.timeout.setObjectName("timeout")
        self.horizontalLayout_4.addWidget(self.timeout)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.total_points = QtWidgets.QLabel(Form)
        self.total_points.setMinimumSize(QtCore.QSize(30, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.total_points.setFont(font)
        self.total_points.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.total_points.setAlignment(QtCore.Qt.AlignCenter)
        self.total_points.setObjectName("total_points")
        self.horizontalLayout.addWidget(self.total_points)
        self.points = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.points.setFont(font)
        self.points.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.points.setAlignment(QtCore.Qt.AlignCenter)
        self.points.setObjectName("points")
        self.horizontalLayout.addWidget(self.points)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.total_num = QtWidgets.QLabel(Form)
        self.total_num.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.total_num.setFont(font)
        self.total_num.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.total_num.setAlignment(QtCore.Qt.AlignCenter)
        self.total_num.setObjectName("total_num")
        self.horizontalLayout_2.addWidget(self.total_num)
        self.num = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.num.setFont(font)
        self.num.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.num.setAlignment(QtCore.Qt.AlignCenter)
        self.num.setObjectName("num")
        self.horizontalLayout_2.addWidget(self.num)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.start = QtWidgets.QPushButton(Form)
        self.start.setMinimumSize(QtCore.QSize(80, 40))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.start.setFont(font)
        self.start.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.start.setObjectName("start")
        self.horizontalLayout_3.addWidget(self.start)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        spacerItem5 = QtWidgets.QSpacerItem(20, 50, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem5)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.horizontalLayout_6.addLayout(self.verticalLayout_4)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.start.clicked.connect(self.onButtonClicked)
        self.timer = QTimer()
        self.timer.timeout.connect(self.my_timer_function)
        self.timer.setInterval(2000)  # 1秒

        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.my_timer_function_2)
        self.timer2.setInterval(100)  # 0.1秒
        self.model_check.stateChanged.connect(self.model_check_changed)
        self.model_check_2.stateChanged.connect(self.model_check_2_changed)

    def model_check_changed(self, state):
        global model_flag
        if state == Qt.Checked:
            # model_check 被选中，取消 model_check_2 的选中状态
            self.model_check_2.setChecked(False)
            model_flag = 1
        else:
            model_flag = 0
        print(model_flag)
    def model_check_2_changed(self, state):
        global model_flag
        if state == Qt.Checked:
            # model_check_2 被选中，取消 model_check 的选中状态
            self.model_check.setChecked(False)
            model_flag = 2
        else:
            model_flag = 0
        print(model_flag)
    # 随机生成数字和颜色
    def generate_random_number_and_color(self):
        global generated_number
        generated_number = random.randint(0, 9)
        color = (0, 255, 0) if random.random() < 0.5 else (0, 0, 255)
        return generated_number, color

    def my_timer_function(self):
        global level, flash_flag, generated_number, generated_color, add_flag, total_cnt, gesture_result
        add_flag = 1
        flash_flag = 0
        level = 0
        gesture_result = None
        generated_number, generated_color = self.generate_random_number_and_color()
        if PRINT_LOG:
            print("number={}".format(generated_number))
        total_cnt += 1
        self.num.setText("{}".format(total_cnt))
        self.points.setText("{}".format(true_cnt))
        self.timer2.start()

    def my_timer_function_2(self):
        global flash_flag, level, generated_time, timer, total_cnt, compare_flag, add_flag, true_cnt, generated_number, \
            gesture_result, generated_color
        self.timer.stop()
        generated_time += Decimal("0.1")
        self.timeout.setText("{}".format(2 - generated_time))
        if compare_flag:
            compare_flag = 0
            if gesture_result != None and gesture_result == generated_number and generated_color == (
                    0, 255, 0) and add_flag == 1:
                if PRINT_LOG:
                    print("result={}".format(gesture_result))
                    print("绿色正确")
                self.timer2.stop()
                if 2 - float(generated_time) >= 1.0:
                    true_cnt += Decimal("1.0")
                    level = 3
                    flash_flag = 1
                elif (2 - float(generated_time)) < 1.0 and (2 - float(generated_time)) > 0.5:
                    true_cnt += Decimal("0.5")
                    level = 2
                    flash_flag = 1
                else:
                    if PRINT_LOG:
                        print("不加分")
                    level = 0
                if PRINT_LOG:
                    print(true_cnt)
                    print("######################{}#################################".format(total_cnt))
                add_flag = 0
                generated_time = Decimal("0.0")
                if total_cnt < 10:
                    self.timer.start()
            elif gesture_result != None and gesture_result != generated_number and generated_color == (
                    0, 0, 255) and add_flag == 1:
                if PRINT_LOG:
                    print("result={}".format(gesture_result))
                    print("红色正确")
                self.timer2.stop()
                if 2 - float(generated_time) >= 1.0:
                    true_cnt += Decimal("1.0")
                    level = 3
                    flash_flag = 1
                elif (2 - float(generated_time)) < 1.0 and (2 - float(generated_time)) > 0.5:
                    true_cnt += Decimal("0.5")
                    level = 2
                    flash_flag = 1
                else:
                    if PRINT_LOG:
                        print("不加分")
                    level = 0
                if PRINT_LOG:
                    print(true_cnt)
                    print("######################{}#################################".format(total_cnt))
                add_flag = 0
                generated_time = Decimal("0.0")
                if total_cnt < 10:
                    self.timer.start()
            else:
                pass

        formatted_time = "{:.1f}".format(2 - float(generated_time))
        if PRINT_LOG:
            print("generated_time: {}".format(formatted_time))
        if 2 - generated_time == 0.0:
            generated_time = Decimal("0.0")
            level = 1
            if PRINT_LOG:
                print("######################{}#################################".format(total_cnt))
            self.timer.stop()
            self.timer.start()
            self.timer2.stop()
            if total_cnt == 10:
                self.timer2.stop()
                self.timer.stop()
                return
        if total_cnt == 10:
            self.timer.stop()
            self.points.setText("{}".format(true_cnt))
            return

    # 这是按钮被点击时会调用的槽函数
    def onButtonClicked(self):
        global gesture_result, generated_number, generated_color, true_cnt, false_cnt, total_cnt, add_flag, generated_time, timer
        if not (self.model_check.isChecked() or self.model_check_2.isChecked()):
            return
        gesture_result = None
        generated_number = None
        generated_color = None
        true_cnt = 0
        false_cnt = 0
        total_cnt = 0
        add_flag = 0
        generated_time = 0
        if generated_number == None:
            self.timer2.stop()
            self.timeout.setText("{}".format(2.0 - generated_time))
        self.timer.start()

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.image.setText(_translate("Form", "规则 \n"
                                              "每局有十轮，每轮2s中，需要在两秒钟内反应出正确的手势，\n"
                                              "反应时间1s内记1分，1-1.5s内记0.5分，1.5-2s内不计分\n"
                                              " 程序会自动生成0-9的手势，数字是绿色的，则比数字对应的手势；\n"
                                              "数字是红色的就比除了这个数字的手势。"))
        self.model_check.setText(_translate("Form", "模型1"))
        self.model_check_2.setText(_translate("Form", "模型2"))
        self.timeout.setText(_translate("Form", "timeout"))
        self.total_points.setText(_translate("Form", "总分"))
        self.points.setText(_translate("Form", "0"))
        self.total_num.setText(_translate("Form", "总个数"))
        self.num.setText(_translate("Form", "0"))
        self.start.setText(_translate("Form", "开始"))


class MyApplication(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video_frame)
        self.ui.start.clicked.connect(self.btn)

    def btn(self):
        if not (self.ui.model_check.isChecked() or self.ui.model_check_2.isChecked()):
            QMessageBox.warning(self, "警告", "请选择一个模型", QMessageBox.Ok)
            return
        print("线程开启")
        self.video_thread.start()

    def update_video_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        self.ui.image.setPixmap(pixmap)

    def closeEvent(self, event):
        hands.close()
        # 停止线程
        if self.video_thread.isRunning():
            self.video_thread.stop()
        
        # 停止定时器
        if self.ui.timer.isActive():
            self.ui.timer.stop()
        if self.ui.timer2.isActive():
            self.ui.timer2.stop()
        
        # 退出应用程序
        QCoreApplication.instance().quit()

        # 调用父类的 closeEvent
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyApplication()
    MainWindow.show()
    sys.exit(app.exec_())
