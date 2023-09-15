# u.py
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import sys
import cv2
import mediapipe as mp
import math
import random
from decimal import Decimal
# 全局变量用于存储手势识别结果和生成的数字

gesture_result = None
generated_number = None
generated_color = None
# generated_time = 0.0
generated_time = Decimal("0.0")
true_cnt = Decimal("0.0")
total_cnt = 0
add_flag = 0
compare_flag = 0





class VideoThread(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()

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
        if 65535. not in angle_list:
            if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                    angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "0"
                gesture_result = 0
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                    angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
                gesture_str = "5"
                gesture_result = 5
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                    angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "8"
                gesture_result = 8
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                    angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
                gesture_str = "7"
                gesture_result = 7
            elif (angle_list[0] > 5) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                    angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "1"
                gesture_result = 1
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                    angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
                gesture_str = "6"
                gesture_result = 6
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (
                    angle_list[2] < thr_angle_s) and (
                    angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
                gesture_str = "3"
                gesture_result = 3
            elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                    angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "9"
                gesture_result = 9
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (
                    angle_list[2] < thr_angle_s) and (
                    angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                gesture_str = "2"
                gesture_result = 2
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (
                    angle_list[2] < thr_angle_s) and (
                    angle_list[3] < thr_angle) and (angle_list[4] < thr_angle):
                gesture_str = "4"
                gesture_result = 4
        return gesture_str

    def run(self):
        global generated_number, gesture_result, generated_color, total_cnt, true_cnt, false_cnt, add_flag, compare_flag
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()  # 读取一帧图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
            frame = cv2.flip(frame, 1)  # 左右镜像
            results = hands.process(frame)  # 将图像传入检测
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB转BGR
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
                # print('gesture_result={}generated_number={}'.format(gesture_result,generated_number))
                compare_flag = 1
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB转BGR
            cv2.putText(frame, str(generated_number), (280, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
            # cv2.putText(frame, str(total_cnt), (400, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
            # cv2.putText(frame, str(true_cnt), (520, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
            # cv2.imshow('MediaPipe Hands', frame)
            self.frame_ready.emit(frame)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 600)
        # 使用setFixedSize方法限制窗口大小
        MainWindow.setFixedSize(900, 600)  # 限制宽度为400像素，高度为300像素
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.vediolabel = QtWidgets.QLabel(self.centralwidget)
        self.vediolabel.setGeometry(QtCore.QRect(50, 50, 600, 480))
        self.vediolabel.setObjectName("vediolabel")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(680, 160, 201, 81))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.sum = QtWidgets.QLabel(self.layoutWidget)
        self.sum.setObjectName("sum")
        self.horizontalLayout.addWidget(self.sum)
        self.total = QtWidgets.QLabel(self.layoutWidget)
        self.total.setObjectName("total")
        self.horizontalLayout.addWidget(self.total)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(680, 250, 201, 71))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.sum_s = QtWidgets.QLabel(self.layoutWidget1)
        self.sum_s.setObjectName("sum_s")
        self.horizontalLayout_2.addWidget(self.sum_s)
        self.s = QtWidgets.QLabel(self.layoutWidget1)
        self.s.setObjectName("s")
        self.horizontalLayout_2.addWidget(self.s)
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(730, 380, 93, 28))
        self.start.setObjectName("start")
        self.time = QtWidgets.QLabel(self.centralwidget)
        self.time.setGeometry(QtCore.QRect(710, 50, 141, 101))
        font = QtGui.QFont()
        font.setPointSize(50)
        self.time.setFont(font)
        self.time.setObjectName("time")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # 创建槽函数
        self.start.clicked.connect(self.onButtonClicked)

        self.timer = QTimer()
        self.timer.timeout.connect(self.my_timer_function)
        self.timer.setInterval(1000)  # 1秒

        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.my_timer_function_2)
        self.timer2.setInterval(100)  # 0.1秒

    # 随机生成数字和颜色
    def generate_random_number_and_color(self):
        global generated_number
        generated_number = random.randint(0, 9)
        color = (0, 255, 0) if random.random() < 0.5 else (0, 0, 255)
        return generated_number, color

    def my_timer_function(self):
        global generated_number, generated_color, add_flag, total_cnt, gesture_result
        add_flag = 1
        generated_number, generated_color = self.generate_random_number_and_color()
        total_cnt += 1
        self.total.setText("{}".format(total_cnt))
        self.s.setText("{}".format(true_cnt))
        self.timer2.start()


    def my_timer_function_2(self):
        global generated_time, timer, total_cnt, compare_flag, add_flag, true_cnt,generated_number,gesture_result,generated_color
        self.timer.stop()
        generated_time += Decimal("0.1")
        self.time.setText("{}".format(2 - generated_time))
        if compare_flag:
            compare_flag = 0
            if gesture_result == generated_number and generated_color == (0, 255, 0) and add_flag == 1:
                print("绿色正确")
                self.timer2.stop()
                if 2 - float(generated_time) > 1.0:
                    true_cnt += Decimal("1.0")
                elif (2 - float(generated_time)) < 1.0 and (2 - float(generated_time)) > 0.5:
                    true_cnt += Decimal("0.5")
                else:
                    print("不加分")
                print(true_cnt)
                print("######################{}#################################".format(total_cnt))
                add_flag = 0
                generated_time = Decimal("0.0")
                if total_cnt < 10:
                    self.timer.start()
                # self.my_timer_function()

            elif gesture_result != generated_number and generated_color == (0, 0, 255) and add_flag == 1:
                print("红色正确")
                self.timer2.stop()
                if 2 - float(generated_time) > 1.0:
                    true_cnt += Decimal("1.0")
                elif (2 - float(generated_time)) < 1.0 and (2 - float(generated_time)) > 0.5:
                    true_cnt += Decimal("0.5")
                else:
                    print("不加分")
                print(true_cnt)
                print("######################{}#################################".format(total_cnt))
                add_flag = 0
                generated_time = Decimal("0.0")
                if total_cnt < 10:
                    self.timer.start()
                # self.my_timer_function()

            else:
                pass

        formatted_time = "{:.1f}".format(2 - float(generated_time))
        print("generated_time: {}".format(formatted_time))
        if 2 - generated_time == 0.0:
            generated_time = Decimal("0.0")
            print("######################{}#################################".format(total_cnt))
            self.timer.start()
            self.timer2.stop()
            if total_cnt == 10:
                self.timer2.stop()
                return
        if total_cnt == 10:
            self.timer.stop()
            self.s.setText("{}".format(true_cnt))
            # generated_number=None
            # self.timer2.stop()
            return


    def onButtonClicked(self):
        global gesture_result, generated_number, generated_color, true_cnt, false_cnt, total_cnt, add_flag, generated_time,timer
        # 这是按钮被点击时会调用的槽函数
        print("Button Clicked!")
        gesture_result = None
        generated_number = None
        generated_color = None
        true_cnt = 0
        false_cnt = 0
        total_cnt = 0
        add_flag = 0
        generated_time = 0
        if generated_number==None:
            self.timer2.stop()
            self.time.setText("{}".format(2.0 - generated_time))
        self.timer.start()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.vediolabel.setText(_translate("MainWindow", "TextLabel"))
        self.sum.setText(_translate("MainWindow", "总个数"))
        self.total.setText(_translate("MainWindow", "0"))
        self.sum_s.setText(_translate("MainWindow", "总得分"))
        self.s.setText(_translate("MainWindow", "0"))
        self.start.setText(_translate("MainWindow", "开始"))
        self.time.setText(_translate("MainWindow", "2.0"))


# Your main application class
class MyApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video_frame)
        self.video_thread.start()

    def update_video_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        self.ui.vediolabel.setPixmap(pixmap)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyApplication()
    MainWindow.show()
    sys.exit(app.exec_())
