import time

import cv2
import numpy as np
import rospy
import mediapipe as mp
import moveit_commander
from geometry_msgs.msg import Pose

arm_group = moveit_commander.MoveGroupCommander("arm")

def calculate_angle(A, B, C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    BA = A - B
    BC = C - B
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)
    angle_radians = np.arccos(cos_angle)
    return angle_radians

def process_hand(hand_landmarks):
    node_0 = [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y,
                hand_landmarks.landmark[0].z]
    node_5 = [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y,
                hand_landmarks.landmark[5].z]
    node_6 = [hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y,
                hand_landmarks.landmark[6].z]
    node_7 = [hand_landmarks.landmark[7].x, hand_landmarks.landmark[7].y,
                hand_landmarks.landmark[7].z]
    node_8 = [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y,
                hand_landmarks.landmark[8].z]
    
    joint_1 = 0
    joint_2 = calculate_angle(node_0, node_5, node_6) - 3.14
    joint_3 = 2.94 - calculate_angle(node_5, node_6, node_7)
    joint_4 = calculate_angle(node_6, node_7, node_8) -1.89
    joint_5 = 0
    return joint_1, joint_2, joint_3, joint_4, joint_5

def move_joint_rad(joint_goal_rad, mode):
    joint_goal = arm_group.get_current_joint_values()
    if mode == 'INC':
        joint_goal[0] += joint_goal_rad[0]
        joint_goal[1] += joint_goal_rad[1]
        joint_goal[2] += joint_goal_rad[2]
        joint_goal[3] += joint_goal_rad[3]
        joint_goal[4] += joint_goal_rad[4]
    elif mode == 'ABS':
        joint_goal[0] = joint_goal_rad[0]
        joint_goal[1] = joint_goal_rad[1]
        joint_goal[2] = joint_goal_rad[2]
        joint_goal[3] = joint_goal_rad[3]
        joint_goal[4] = joint_goal_rad[4]

    arm_group.go(joint_goal, wait=False)
    # time.sleep(0.2)
    # arm_group.stop()
    
rospy.init_node('gesture_node')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)
rospy.loginfo('Mediapipe hands initialized.')

cap = cv2.VideoCapture(0)
rospy.loginfo('Camera initialized.')
tick = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    annotated_image = image.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if tick % 25 == 0:
                joint_1, joint_2, joint_3, joint_4, joint_5 = process_hand(hand_landmarks)
                move_joint_rad([joint_1, joint_2, joint_3, joint_4, joint_5], 'ABS')
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', annotated_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    else:
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    tick += 1

cv2.destroyAllWindows()
cap.release()
rospy.loginfo('Camera released.')
