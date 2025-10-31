import sys
import numpy as np
import serial
import cv2
import mediapipe as mp
import threading
import csv
import time
from datetime import datetime
import os
# import pprint
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QTextBrowser, QVBoxLayout, QPushButton, QSizePolicy
from PySide6.QtGui import QMatrix4x4, QImage, QPixmap
from PySide6.QtCore import QTimer
import pyqtgraph.opengl as gl
import torch
from pathlib import Path

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
widget_size = 400, 300
log_file_path = Path(r"logs")
ui_file_path = Path(r'gui/ui/MonitorForm.ui')
Gyro_COMport='COM3'  # 替換為實際的 COM 埠
Gyro_baudrate=115200

class Gyro3DApp(QWidget):

    def __init__(self, port=Gyro_COMport, baudrate=Gyro_baudrate):
        super().__init__()
        
        # Load the UI file
        loader = QUiLoader()
        ui = loader.load(ui_file_path, self)
        
        # Set window title
        self.setWindowTitle("訓練數據獲取監看視窗")

        # 在 __init__ 加入：
        self.log_data = []  # 儲存暫時資料
        self.raw_frames = []  # 用來儲存原始影像
        self.raw_hands_frames = []  # 用來儲存原始影像
        self.logging = False  # 是否記錄
        self.last_log_time = time.time()

        # 取得 QLineEdit 元件
        self.lineEdit_path = ui.findChild(QLineEdit, "lineEdit")
        self.lineEdit_folder = ui.findChild(QLineEdit, "lineEdit_2")

        # 預設值
        self.lineEdit_path.setText(log_file_path)
        self.lineEdit_folder.setText(datetime.now().strftime("%Y%m%d"))


        # 建立 3D 視圖
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=10)
        self.gl_widget.setFixedSize(*widget_size)

        # 建立 3D 坐標軸
        axis = gl.GLAxisItem()
        self.gl_widget.addItem(axis)

        # 建立 3D 立方體
        self.cube = self.create_cube()
        self.gl_widget.addItem(self.cube)

        # 更新標籤以顯示陀螺儀數值
        self.yaw_label = ui.findChild(QLabel, "label")
        self.pitch_label = ui.findChild(QLabel, "label_2")
        self.roll_label = ui.findChild(QLabel, "label_3")

        # 為重置按鈕連接事件
        self.pushButton = ui.findChild(QPushButton, "pushButton")
        self.pushButton.clicked.connect(self.reset_values)

        # 新增開始/停止記錄按鈕
        self.toggle_logging_button = ui.findChild(QPushButton, "pushButton_2")
        self.toggle_first_click = True
        self.toggle_logging_button.clicked.connect(self.toggle_logging)

        # 更新紀錄狀態顯示標籤
        self.recording_status_label = ui.findChild(QLabel, "label_5")

        ## 更新檔案路徑顯示標籤
        self.file_path_label = ui.findChild(QLabel, "label_6")
        
        # 新增 CUDA 和 cuDNN 狀態標籤
        self.cuda_status_label = ui.findChild(QLabel, "label_8")
        cuda_available = torch.cuda.is_available()
        cudnn_available = torch.backends.cudnn.is_available()
        self.cuda_status_label.setText(f"CUDA: {'可用' if cuda_available else '不可用'}, cuDNN: {'可用' if cudnn_available else '不可用'}")

        # 將 GLViewWidget 添加到新的 UI 容器
        self.openGLWidget = ui.findChild(QWidget, "openGLWidget")
        if self.openGLWidget.layout() is None:
            self.openGLWidget.setLayout(QVBoxLayout())
        self.openGLWidget.layout().addWidget(self.gl_widget)

        # 更新手部座標顯示標籤
        self.label_4 = ui.findChild(QTextBrowser, "textBrowser")
        # 設定 label_4 的大小策略，防止其變大
        self.label_4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # 設置串口通信
        self.serial_port = serial.Serial(port, baudrate, timeout=1)
        self.running = True

        # 記錄偏移量
        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.roll_offset = 0.0

        # 開啟讀取串口的執行緒
        self.thread = threading.Thread(target=self.read_serial)
        self.thread.start()

        # 加入 Mediapipe 手部偵測影像視圖
        self.hand_widget = QLabel(self)
        self.hand_widget.setFixedSize(*widget_size)

        # 將手部偵測影像視圖添加到新的 UI 容器
        self.widget = ui.findChild(QWidget, "widget")
        if self.widget.layout() is None:
            self.widget.setLayout(QVBoxLayout())
        self.widget.layout().addWidget(self.hand_widget)

        # 設置定時器來模擬數據
        self.timer = QTimer()
        # self.timer.timeout.connect(self.generate_fake_data)
        # self.timer.start(100)

        # 設定攝影機與 Mediapipe
        self.cap = cv2.VideoCapture(0)
        # 修改 Mediapipe 手部偵測設定，限制最多一隻手
        self.hand_detector = mp_hands.Hands(min_detection_confidence=0.5, 
                                           min_tracking_confidence=0.5, 
                                           max_num_hands=1)  # 限制最多一隻手
        self.hand_timer = QTimer()
        self.hand_timer.timeout.connect(self.process_hand_detection)
        self.hand_timer.start(30)

        

    def create_cube(self):
        verts = np.array([
            [-2, -2,  2], [2, -2,  2],[0.5, 0.5,  0.5], [-0.5, 0.5,  0.5],
            [-2, -2, -2], [2, -2, -2],[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]
        ])
        faces = np.array([
            [0, 1, 2], [2, 3, 0], [4, 5, 6], [6, 7, 4],
            [0, 1, 5], [5, 4, 0], [2, 3, 7], [7, 6, 2],
            [0, 3, 7], [7, 4, 0], [1, 2, 6], [6, 5, 1]
        ])
        mesh = gl.GLMeshItem(vertexes=verts, faces=faces, color=(0, 1, 0, 0.5), drawEdges=True, drawFaces=True)
        return mesh
    
    def read_serial(self):
        while self.running:
            try:
                line = self.serial_port.readline().decode().strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        yaw, pitch, roll = map(float, parts)
                        yaw -= self.yaw_offset
                        pitch = -pitch - self.pitch_offset
                        roll = -roll - self.roll_offset
                        self.update_cube(yaw, pitch, roll)
                        self.update_labels(yaw, pitch, roll)
            except Exception as e:
                print("Error:", e)
                break

    
    def update_cube(self, yaw, pitch, roll):
        rot_matrix = QMatrix4x4()
        rot_matrix.rotate(yaw, 0, 0, 1)
        rot_matrix.rotate(pitch, 1, 0, 0)
        rot_matrix.rotate(roll, 0, 1, 0)
        self.cube.setTransform(rot_matrix)

    def update_labels(self, yaw, pitch, roll):
        self.yaw_label.setText(f"Yaw: {yaw:.2f}")
        self.pitch_label.setText(f"Pitch: {pitch:.2f}")
        self.roll_label.setText(f"Roll: {roll:.2f}")

    def reset_values(self):
        # 使用按鈕重置偏移量
        self.yaw_offset += float(self.yaw_label.text().split(': ')[1])
        self.pitch_offset += float(self.pitch_label.text().split(': ')[1])
        self.roll_offset += float(self.roll_label.text().split(': ')[1])
        
        # # 禁用按鈕
        # self.pushButton.setEnabled(False)

    def reset_gyro_data(self):
        '''
        # === 修改開始：發送 'R' 指令給 Arduino 要求重置 ===
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(b'R')  # 傳送單一字元 'R'
                # 稍微等一下 Arduino 執行 resetGyro()
                QTimer.singleShot(300, self.reset_values)  # 延遲 300ms 執行 offset 計算
            except Exception as e:
                print("Serial write error:", e)
        # === 修改結束 ===
        '''
        
       
        self.update_cube(self.yaw, self.pitch, self.roll)
        self.update_labels(self.yaw, self.pitch, self.roll)
        

        
    def toggle_logging(self):
        # 切換記錄狀態
        self.logging = not self.logging
        if self.toggle_first_click:
            self.toggle_first_click = False  # 更新狀態為已按下
            # 設定 QLineEdit 為唯讀模式
            self.lineEdit_path.setReadOnly(True)
            self.lineEdit_folder.setReadOnly(True)
            # 指定存放位置
            log_dir = os.path.join(self.lineEdit_path.text(), self.lineEdit_folder.text())
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)  # 自動建立資料夾

            # 設定檔案名稱
            # 自動生成遞增的檔案名稱
            file_index = 1
            while True:
                self.csv_filename = os.path.join(log_dir, f"log_{file_index:02d}.csv")
                if not os.path.exists(self.csv_filename):  # 如果檔案不存在，使用該名稱
                    break
                file_index += 1
            self.file_path_label.setText(f"檔案路徑: {self.csv_filename}")

        if self.logging:
            self.toggle_logging_button.setText("停止記錄")
            # self.recording_status_label.setText("Recording")
            # self.recording_status_label.setStyleSheet("background-color: red; color: white;")  # 紅色背景

        else:
            self.toggle_logging_button.setText("開始記錄")
            self.recording_status_label.setText("Preparing")
            self.recording_status_label.setStyleSheet("background-color: black; color: white;")  # 綠色背景

    def process_hand_detection(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 儲存原始 BGR 影像
        if self.logging:
            self.raw_frames.append(frame.copy())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換顏色格式
        frame_widget = frame

        # 取得 QLabel 的大小，確保影像適應 widget
        widget_size = self.hand_widget.size()
        frame_widget = cv2.resize(frame_widget, (widget_size.width(), widget_size.height()))

        results = self.hand_detector.process(frame_widget)

        hand_landmarks_text = "手部座標:\n"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame_widget, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 取得座標並轉換成影像空間
                for idx, lm in enumerate(hand_landmarks.landmark):
                    x, y = lm.x,lm.y
                    hand_landmarks_text += f"點 {idx}: ({x}, {y})\n"

        # 更新 QLabel 以顯示座標
        self.label_4.setText(hand_landmarks_text)

        # 顯示影像
        h, w, c = frame_widget.shape
        qimg = QImage(frame_widget.data, w, h, 3 * w, QImage.Format_RGB888)
        self.hand_widget.setPixmap(QPixmap.fromImage(qimg))

        if self.logging:
            now = time.time()
            if now - self.last_log_time >= 1/60:  # 60 FPS
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                # 收集資料
                row = [timestamp, self.yaw_label.text(), self.pitch_label.text(), self.roll_label.text()]
                # 收集手部座標
                if results.multi_hand_landmarks:
                    self.raw_hands_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy())  # 儲存原始影像
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            row.extend([lm.x, lm.y, lm.z])
                    self.log_data.append(row)
                    self.last_log_time = now
                    self.recording_status_label.setText("Recording")
                    self.recording_status_label.setStyleSheet("background-color: red; color: white;")  # 紅色背景
                else:
                    self.recording_status_label.setText("Stopped")
                    self.recording_status_label.setStyleSheet("background-color: green; color: white;")  # 綠色背景
                    #row.extend([None] * 63)  # 若沒偵測到手，補空值
                    
    def closeEvent(self, event):
        # 儲存 CSV 檔案
        if self.log_data:
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 建立標頭
                header = ['Timestamp', 'Yaw', 'Pitch', 'Roll']
                for i in range(21):
                    header += [f'X{i}', f'Y{i}', f'Z{i}']
                writer.writerow(header)
                writer.writerows(self.log_data)
            print(f"紀錄完成：{self.csv_filename}")

        # 儲存片段影像為影片
        if self.raw_hands_frames:
            video_path = self.csv_filename.replace('.csv', '.avi')
            # 在檔名加上 H 前綴
            dir_name, base_name = os.path.split(video_path)
            video_path = os.path.join(dir_name, 'H' + base_name)
            h, w, _ = self.raw_hands_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
            for frame in self.raw_hands_frames:
                out.write(frame)
            out.release()
            print(f"片段影像已儲存：{video_path}")

        # 儲存原始影像為影片
        if self.raw_frames:
            video_path = self.csv_filename.replace('.csv', '.avi')
            h, w, _ = self.raw_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
            for frame in self.raw_frames:
                out.write(frame)
            out.release()
            print(f"原始影像已儲存：{video_path}")

        self.timer.stop()
        self.hand_timer.stop()
        self.cap.release()
        self.running = False  # 停止執行緒
        if self.thread.is_alive():
            self.thread.join()  # 等待執行緒結束
        self.serial_port.close()  # 關閉串口

        # 調用父類的 closeEvent 方法
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Gyro3DApp()
    window.show()
    sys.exit(app.exec())