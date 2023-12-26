import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import pyautogui
import time
import keyboard

# Khởi tạo trạng thái dự đoán và điều khiển chuột
pause = True

# Khởi tạo biến đếm cho mỗi hướng và tốc độ di chuyển chuột ban đầu
direction_counts = {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'center': 0, 'blink': 0}
mouse_speeds = {'up': 10, 'down': 10, 'left': 10, 'right': 10, 'center': 10, 'blink': 10}

# Đặt biến đếm khung hình và thời gian bắt đầu
frame_count = 0
start_time = time.time()

# Load mô hình đã được huấn luyện
model_path = 'C:/tgmt_ytb1/model_cnn_24_12_full_model_main_eye_add_csv_graph.h5'
model_path2 = 'C:/tgmt_ytb1/model_24_12_only_right_eye.h5'

model = tf.keras.models.load_model(model_path)
model2 = tf.keras.models.load_model(model_path2)

# Định nghĩa các nhãn của hướng nhìn
labels = ['left', 'right', 'up', 'down', 'center', 'blink']
labels2 = ['blink', 'noblink']

# Khởi tạo bộ phân loại landmask
face_mesh = mp.solutions.face_mesh.FaceMesh()

# Mở webcam
cap = cv2.VideoCapture(0)

# Khởi tạo biến đếm số frame và số frame bỏ qua
frame_count2 = 0
skip_frame = 3  # 3 là tối ưu nhất

while True:
    if cv2.waitKey(1) & 0xFF == ord('p'):
        pause = not pause
        time.sleep(1)  # Ngủ 1 giây để tránh nhận nhiều lần khi giữ nút 'p'

    if keyboard.is_pressed('q'):
        break

    if pause:
        # Hiển thị video và fps khi ở trạng thái pause
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Tính toán FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > 1.0:  # Cập nhật FPS mỗi giây
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Đặt lại biến thời gian bắt đầu và frame_count
            start_time = time.time()
            frame_count = 0
        # Hiển thị FPS lên khung hình
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Frame', frame)

    else:
        # Đọc khung hình từ webcam
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Chuyển đổi khung hình sang không gian màu RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Dò tìm landmask trên khung hình
        results = face_mesh.process(frame_rgb)

        # Trích xuất danh sách các điểm ảnh landmask
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x1 = int(face_landmarks.landmark[27].x * frame.shape[1])
                y1 = int(face_landmarks.landmark[27].y * frame.shape[0])

                x2 = int(face_landmarks.landmark[23].x * frame.shape[1])
                y2 = int(face_landmarks.landmark[23].y * frame.shape[0])

                x3 = int(face_landmarks.landmark[243].x * frame.shape[1])
                y3 = int(face_landmarks.landmark[243].y * frame.shape[0])

                x4 = int(face_landmarks.landmark[130].x * frame.shape[1])
                y4 = int(face_landmarks.landmark[130].y * frame.shape[0])

                x_min = min(x1, x2, x3, x4)
                x_max = max(x1, x2, x3, x4)
                y_min = min(y1, y2, y3, y4)
                y_max = max(y1, y2, y3, y4)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                a1 = int(face_landmarks.landmark[257].x * frame.shape[1])
                b1 = int(face_landmarks.landmark[257].y * frame.shape[0])

                a2 = int(face_landmarks.landmark[253].x * frame.shape[1])
                b2 = int(face_landmarks.landmark[253].y * frame.shape[0])

                a3 = int(face_landmarks.landmark[463].x * frame.shape[1])
                b3 = int(face_landmarks.landmark[463].y * frame.shape[0])

                a4 = int(face_landmarks.landmark[359].x * frame.shape[1])
                b4 = int(face_landmarks.landmark[359].y * frame.shape[0])

                a_min = min(a1, a2, a3, a4)
                a_max = max(a1, a2, a3, a4)
                b_min = min(b1, b2, b3, b4)
                b_max = max(b1, b2, b3, b4)
                # cv2.rectangle(frame, (a_min, b_min), (a_max, b_max), (0, 255, 0), 2)

                cropped_image = frame[y_min:y_max, x_min:x_max]  # Mắt trái
                cropped_image2 = frame[b_min:b_max, a_min:a_max]  # Mắt phải
                if frame_count2 % skip_frame == 0:
                    if cropped_image is not None and cropped_image.size > 0:
                        resized_image = cv2.resize(cropped_image, (64, 64))
                        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                        image = gray_image.astype('float32') / 255.0

                        # Thêm một chiều mới vào đầu và cuối mảng
                        input_image = np.expand_dims(image, axis=-1)
                        input_image = np.reshape(input_image, (1, 64, 64, 1))

                        # Dự đoán hướng nhìn từ ảnh cắt
                        predictions = model.predict(input_image)
                        predicted_label = labels[np.argmax(predictions)]
                        cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if cropped_image2 is not None and cropped_image2.size > 0:
                        resized_image2 = cv2.resize(cropped_image2, (64, 64))
                        gray_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)
                        image2 = gray_image2.astype('float32') / 255.0

                        # Thêm một chiều mới vào đầu và cuối mảng
                        input_image2 = np.expand_dims(image2, axis=-1)
                        input_image2 = np.reshape(input_image2, (1, 64, 64, 1))

                    if cropped_image is not None and cropped_image.size > 0 and cropped_image2 is not None and cropped_image2.size > 0:
                        if predicted_label != 'blink':
                            for direction in direction_counts:
                                if predicted_label == direction:
                                    direction_counts[direction] += 1
                                    if direction_counts[direction] >= 2:
                                        mouse_speeds[direction] += 10
                                        direction_counts[direction] = 0
                                else:
                                    direction_counts[direction] = 0
                                    mouse_speeds[direction] = 10

                            if predicted_label == 'up':
                                pyautogui.moveRel(0, -mouse_speeds['up'], duration=0.0001)
                            elif predicted_label == 'down':
                                pyautogui.moveRel(0, mouse_speeds['down'], duration=0.0001)
                            elif predicted_label == 'left':
                                pyautogui.moveRel(-mouse_speeds['left'], 0, duration=0.0001)
                            elif predicted_label == 'right':
                                pyautogui.moveRel(mouse_speeds['right'], 0, duration=0.0001)
                            elif predicted_label == 'center':
                                pyautogui.moveRel(0, 0, duration=0.0001)
                        if predicted_label == 'blink':
                            predictions2 = model2.predict(input_image2)
                            predicted_label2 = labels2[np.argmax(predictions2)]
                            # cv2.putText(frame, predicted_label2, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 255, 0), 2)
                            if predicted_label2 != 'blink':
                                pyautogui.click()
                else:
                    pass
                frame_count2 += 1
                cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                            2)

        # Tính toán FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > 1.0:  # Cập nhật FPS mỗi giây
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Đặt lại biến thời gian bắt đầu và frame_count
            start_time = time.time()
            frame_count = 0
        # Hiển thị FPS lên khung hình
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Hiển thị khung hình
        cv2.imshow('Frame', frame)

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
