import sys
import numpy as np
import serial
import cv2
import mediapipe as mp
import threading
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QMatrix4x4, QImage, QPixmap
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
widget_size = 320, 240

class Gyro3DApp(QWidget):

    def __init__(self, port='COM4', baudrate=115200):
        super().__init__()
        
        # Load the UI file
        uic.loadUi(r'手部控制平台\視窗程式\ui\MonitorForm.ui', self)
        
        # Set window title
        self.setWindowTitle("訓練數據獲取監看視窗")

        # 設定 widget 大小


        # 建立 3D 視圖
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=10)
        self.gl_widget.setFixedSize(*widget_size)
        
        # 將 GLViewWidget 添加到 UI 中的 widget 容器
        if self.widget.layout() is None:
            self.widget.setLayout(QVBoxLayout())
        self.widget.layout().addWidget(self.gl_widget)

        # 建立 3D 坐標軸
        axis = gl.GLAxisItem()
        self.gl_widget.addItem(axis)

        # 建立 3D 立方體
        self.cube = self.create_cube()
        self.gl_widget.addItem(self.cube)

        # 更新標籤以顯示陀螺儀數值
        self.yaw_label = self.label
        self.pitch_label = self.label_2
        self.roll_label = self.label_3

        # 為按鈕連接事件
        self.pushButton.clicked.connect(self.reset_values)

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
        
        if self.widget_2.layout() is None:
            self.widget_2.setLayout(QVBoxLayout())
        self.widget_2.layout().addWidget(self.hand_widget)

        # 設置定時器來模擬數據
        self.timer = QTimer()
        #self.timer.timeout.connect(self.generate_fake_data)
        #self.timer.start(100)
        
        # 設定攝影機與 Mediapipe
        self.cap = cv2.VideoCapture(0)
        self.hand_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hand_timer = QTimer()
        self.hand_timer.timeout.connect(self.process_hand_detection)
        self.hand_timer.start(30)

    def create_cube(self):
        verts = np.array([
            [-2, -2, -2], [2, -2, -2], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-2, -2,  2], [2, -2,  2], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5]
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
        self.yaw_offset = float(self.yaw_label.text().split(': ')[1])
        self.pitch_offset = float(self.pitch_label.text().split(': ')[1])
        self.roll_offset = float(self.roll_label.text().split(': ')[1])
        
        # 禁用按鈕
        self.pushButton.setEnabled(False)

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
        

        
    
    def process_hand_detection(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換顏色格式

        # 取得 QLabel 的大小，確保影像適應 widget
        widget_size = self.hand_widget.size()
        frame = cv2.resize(frame, (widget_size.width(), widget_size.height()))

        results = self.hand_detector.process(frame)

        hand_landmarks_text = "手部座標:\n"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 取得座標並轉換成影像空間
                for idx, lm in enumerate(hand_landmarks.landmark):
                    x, y = int(lm.x * widget_size.width()), int(lm.y * widget_size.height())
                    hand_landmarks_text += f"點 {idx}: ({x}, {y})\n"

        # 更新 QLabel 以顯示座標
        self.label_4.setText(hand_landmarks_text)

        # 顯示影像
        h, w, c = frame.shape
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
        self.hand_widget.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self.timer.stop()
        self.hand_timer.stop()
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Gyro3DApp(port='COM4')
    window.show()
    sys.exit(app.exec_())