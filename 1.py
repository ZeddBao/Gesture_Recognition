#!/usr/bin/python3
import random

import cv2
import mediapipe as mp
import math
import time
import threading

# 全局变量用于存储手势识别结果和生成的数字
gesture_result = None
generated_number = None
generated_color = None
true_cnt = 0
false_cnt = 0
total_cnt = 0
add_flag = 0


def vector_2d_angle(v1, v2):
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


def hand_angle(hand_):
    '''
        获取对应手相关向量的二维角度,根据角度确定手势
    '''
    angle_list = []
    # ---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    return angle_list


def h_gesture(angle_list):
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
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
            gesture_str = "3"
            gesture_result = 3
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "9"
            gesture_result = 9
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "2"
            gesture_result = 2
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle) and (angle_list[4] < thr_angle):
            gesture_str = "4"
            gesture_result = 4
    return gesture_str


# 随机生成数字和颜色
def generate_random_number_and_color():
    global generated_number
    generated_number = random.randint(0, 9)
    color = (0, 255, 0) if random.random() < 0.5 else (0, 0, 255)
    return generated_number, color


def my_timer_function():
    global generated_number, generated_color, add_flag, total_cnt, gesture_result
    # print(time.time())
    add_flag = 1
    total_cnt += 1
    generated_number, generated_color = generate_random_number_and_color()
    print(generated_color)

    timer = threading.Timer(2, my_timer_function)
    if total_cnt == 10:
        timer.cancel()
        return
    timer.start()


def detect():
    global generated_number, gesture_result, generated_color, total_cnt, true_cnt, false_cnt, add_flag
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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB转BGR
        time1 = time.time()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame.shape[1]
                    y = hand_landmarks.landmark[i].y * frame.shape[0]
                    hand_local.append((x, y))
                if hand_local:
                    angle_list = hand_angle(hand_local)
                    gesture_str = h_gesture(angle_list)
                    cv2.putText(frame, gesture_str, (0, 100), 0, 1.3, (0, 0, 255), 3)
            # print('gesture_result={}generated_number={}'.format(gesture_result,generated_number))
            if gesture_result == generated_number and generated_color == (0, 255, 0) and add_flag == 1:
                # total_cnt += 1
                print("绿色正确")
                true_cnt += 1
                add_flag = 0
            elif gesture_result != generated_number and generated_color == (0, 0, 255) and add_flag == 1:
                # total_cnt += 1
                print("红色正确")
                true_cnt += 1
                add_flag = 0
            else:
                pass
        cv2.putText(frame, str(generated_number), (280, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
        cv2.putText(frame, str(total_cnt), (400, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
        cv2.putText(frame, str(true_cnt), (520, 100), cv2.FONT_HERSHEY_PLAIN, 5, generated_color, 10)
        cv2.imshow('MediaPipe Hands', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # or total_cnt==10
            cap.release()
            break


if __name__ == '__main__':
    # 创建一个5秒后触发的定时器，指定要执行的函数
    timer = threading.Timer(2, my_timer_function)
    timer.start()
    cv2.namedWindow('MediaPipe Hands')
    detect()
