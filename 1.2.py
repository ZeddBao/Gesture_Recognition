import threading
import cv2
import mediapipe as mp
import random
import numpy as np
import math
import queue
import time
# 初始化MediaPipe手势识别
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

# 创建队列用于线程之间的通信
queue_lock = threading.Lock()
gesture_queue = queue.Queue()
number_queue = queue.Queue()

# 全局变量用于存储手势识别结果和生成的数字
gesture_result = None
generated_number = None



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
            gesture_str = "fist"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
            gesture_str = "five"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "gun"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
            gesture_str = "love"
        elif (angle_list[0] > 5) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "one"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
            gesture_str = "six"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
            gesture_str = "three"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "thumbUp"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "two"
    return gesture_str


# 随机生成数字和颜色
def generate_random_number_and_color():
    global generated_number
    generated_number = random.randint(0, 9)
    color = (0, 255, 0) if random.random() < 0.5 else (0, 0, 255)
    return generated_number, color


# 线程1：手势识别
def gesture_recognition():
    global gesture_result
    cap = cv2.VideoCapture(0)  # 你可以根据需要更改摄像头索引
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
        frame = cv2.flip(frame, 1)  # 左右镜像
        results = hands.process(frame)  # 将图像传入检测
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB转BGR

        if results.multi_hand_landmarks:
            # 这里可以处理识别到的手势信息
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
                    # 将手势信息放入队列
                    with queue_lock:
                        gesture_queue.put(gesture_str)
            gesture_result = results.multi_hand_landmarks
        else:
            gesture_result = None

        # 实时显示手势识别的图像
        cv2.imshow('MediaPipe Hands', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 线程2：生成数字
def generate_numbers():
    global generated_number
    for _ in range(10):  # 进行十轮
        number, color = generate_random_number_and_color()
        # 将生成的数字和颜色放入队列
        with queue_lock:
            number_queue.put((number, color))
        # 等待5秒
        time.sleep(5)

# 线程3：显示生成的数字
def display_generated_numbers():
    while True:
        try:
            number, color = number_queue.get(timeout=1)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, str(number), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)
            cv2.imshow("Generated Number", frame)
            cv2.waitKey(5000)  # 显示数字5秒
            cv2.imshow("Generated Number", np.zeros((480, 640, 3), dtype=np.uint8))  # 清除生成的数字
            cv2.waitKey(1000)  # 等待1秒
        except queue.Empty:
            continue

if __name__ == "__main__":
    # 创建线程1、线程2和线程3
    thread1 = threading.Thread(target=gesture_recognition)
    thread2 = threading.Thread(target=generate_numbers)
    thread3 = threading.Thread(target=display_generated_numbers)

    # 启动线程1、线程2和线程3
    thread1.start()
    thread2.start()
    thread3.start()

    # 主线程等待线程1、线程2和线程3结束
    thread1.join()
    thread2.join()
    thread3.join()
