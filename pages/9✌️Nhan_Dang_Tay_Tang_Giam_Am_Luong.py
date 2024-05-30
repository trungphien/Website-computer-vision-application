import cv2
import time
import os
import utility.B9_RecognitionFinger.hand as htm
import streamlit as st
import pyautogui
import math


st.set_page_config(
    page_title="Nhận Dạng Tay Tăng Giảm Âm Lượng",
    page_icon="✌️",
)
# Function to adjust volume based on finger distance
def adjust_volume(finger_distance):
    # Chọn một ngưỡng khoảng cách phù hợp để điều chỉnh âm lượng
    threshold_distance = 50  # Tùy chỉnh giá trị này để phù hợp với ứng dụng của bạn
    
    if finger_distance > threshold_distance:
        pyautogui.press('volumeup')
    else:
        pyautogui.press('volumedown')

# Main Streamlit app
st.sidebar.markdown("Nhận Dạng Tay")

def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():
        if val == key:
            return value

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page', ['Finger Recognition'])
if app_mode == 'Finger Recognition':
    st.title("Finger Recognition")

    start_btn = st.button('START')
    exit_btn = st.button('EXIT')

    if start_btn:
        cap = cv2.VideoCapture(0)

        FRAME_WINDOW = st.image([])

        FolderPath = "utility/B9_RecognitionFinger/Fingers"
        lst = os.listdir(FolderPath)

        lst_2 = []  # khai báo list chứa các mảng giá trị của các hình ảnh/
        for i in lst:
            image = cv2.imread(f"{FolderPath}/{i}")  # Fingers/1.jpg , Fingers/2.jpg ...
            lst_2.append(image)

        pTime = 0
        detector = htm.handDetector(detectionCon=1)
        fingerid = [4, 8, 12, 16, 20]

        while True:
            ret, frame = cap.read()
            frame = detector.findHands(frame)
            lmList = detector.findPosition(frame, draw=False)  # phát hiện vị trí

            if len(lmList) != 0:
                fingers = []

                # Thumb
                if lmList[fingerid[0]][1] < lmList[fingerid[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Index finger
                if lmList[fingerid[1]][1] < lmList[fingerid[1] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                print(fingers)

                # Tính khoảng cách giữa ngón cái và ngón trỏ
                thumb_tip = (lmList[fingerid[0]][1], lmList[fingerid[0]][2])
                index_tip = (lmList[fingerid[1]][1], lmList[fingerid[1]][2])

                finger_distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
                print(f"Finger distance: {finger_distance}")

                # Chỉnh âm lượng dựa trên khoảng cách giữa ngón cái và ngón trỏ
                adjust_volume(finger_distance)

                h, w, c = lst_2[fingers.count(1) - 1].shape
                frame[0:h, 0:w] = lst_2[fingers.count(1) - 1]
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame, f"FPS: {int(fps)}", (150, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            print(frame)
            FRAME_WINDOW.image(frame)
            if exit_btn:
                break