import sys
import cv2
import sqlite3
import numpy as np
import mediapipe as mp
import pyttsx3
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QStackedWidget, QMessageBox, QComboBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QFrame, QSizePolicy)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtTextToSpeech import QTextToSpeech
from PyQt6.QtCore import QLocale

DB_NAME = 'telemedycyna.db'

# ==========================================
STYLE_SHEET = """
QMainWindow {
    background-color: #f0f4f8;
}
QLabel {
    font-size: 14px;
    color: #333333;
}
QLabel#header {
    font-size: 26px;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 20px;
}
QLabel#video_feed {
    background-color: #1e1e1e;
    color: #ffffff;
    border-radius: 10px;
}
QLineEdit {
    padding: 12px;
    font-size: 14px;
    border: 1px solid #ced4da;
    border-radius: 6px;
    background-color: #ffffff;
}
QLineEdit:focus {
    border: 2px solid #3498db;
}
QPushButton {
    background-color: #3498db;
    color: white;
    padding: 12px 20px;
    font-size: 14px;
    border: none;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2980b9;
}
QPushButton:disabled {
    background-color: #95a5a6;
    color: #ecf0f1;
}
QPushButton#danger {
    background-color: #e74c3c;
}
QPushButton#danger:hover {
    background-color: #c0392b;
}
QComboBox {
    padding: 10px;
    font-size: 14px;
    border: 1px solid #ced4da;
    border-radius: 6px;
    background-color: #ffffff;
}
QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f9f9f9;
    border: 1px solid #ced4da;
    border-radius: 6px;
    font-size: 14px;
}
QHeaderView::section {
    background-color: #ecf0f1;
    padding: 8px;
    font-weight: bold;
    border: none;
    border-bottom: 2px solid #bdc3c7;
}
QFrame#login_box {
    background-color: #ffffff;
    border-radius: 10px;
    border: 1px solid #e1e8ed;
}
"""


# ==========================================
# 1. BAZA DANYCH (SQLite)
# ==========================================
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users
                          (
                              id
                              INTEGER
                              PRIMARY
                              KEY,
                              username
                              TEXT,
                              password
                              TEXT,
                              role
                              TEXT
                          )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS tests
                          (
                              id
                              INTEGER
                              PRIMARY
                              KEY,
                              patient_username
                              TEXT,
                              result_data
                              TEXT,
                              doctor_decision
                              TEXT
                          )''')

        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                           ('pacjent1', '123', 'pacjent'))
            cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                           ('lekarz1', '123', 'lekarz'))


# ==========================================
# 2. WĄTKI POBOCZNE (Audio i Wideo)
# ==========================================
class CameraMediaPipeThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    test_result_signal = pyqtSignal(str)

    def __init__(self, camera_id=0, test_type='right_arm'):
        super().__init__()
        self.camera_id = camera_id
        self.test_type = test_type
        self._run_flag = True
        self.test_passed = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Błąd: Nie można otworzyć kamery o ID {self.camera_id}.")
            return

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self._run_flag:
                ret, frame = cap.read()
                if ret:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = pose.process(image_rgb)
                    image_rgb.flags.writeable = True

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        # --- DIAGNOSTIC LOGIC ---
                        if not self.test_passed:
                            landmarks = results.pose_landmarks.landmark

                            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                            r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

                            r_visible = r_shoulder.visibility > 0.5 and r_wrist.visibility > 0.5
                            l_visible = l_shoulder.visibility > 0.5 and l_wrist.visibility > 0.5
                            passed = False

                            if self.test_type == 'right_arm' and r_visible:
                                if r_wrist.y < r_shoulder.y: passed = True
                            elif self.test_type == 'left_arm' and l_visible:
                                if l_wrist.y < l_shoulder.y: passed = True
                            elif self.test_type == 'both_arms' and r_visible and l_visible:
                                if r_wrist.y < r_shoulder.y and l_wrist.y < l_shoulder.y: passed = True
                            elif self.test_type == 'hands_together' and r_visible and l_visible:
                                dist = math.hypot(r_wrist.x - l_wrist.x, r_wrist.y - l_wrist.y)
                                if dist < 0.05: passed = True

                            if passed:
                                self.test_passed = True
                                self.test_result_signal.emit("Sukces: Zadanie wykonane poprawnie.")

                    image_rgb = np.ascontiguousarray(image_rgb)
                    h, w, ch = image_rgb.shape
                    bytes_per_line = ch * w

                    q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                    self.change_pixmap_signal.emit(q_img)

                QThread.msleep(30)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# ==========================================
# 3. GŁÓWNE OKNA APLIKACJI (GUI)
# ==========================================
class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tts = QTextToSpeech()
        self.tts.setLocale(QLocale(QLocale.Language.Polish))
        self.setWindowTitle("System Diagnostyki Neurologicznej")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(STYLE_SHEET)

        self.current_user = None

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.init_login_screen()
        self.init_patient_screen()
        self.init_doctor_screen()

        self.stacked_widget.setCurrentIndex(0)

    def init_login_screen(self):
        widget = QWidget()
        main_layout = QVBoxLayout()

        # Centered container for login to prevent stretching
        login_container = QFrame()
        login_container.setObjectName("login_box")
        login_container.setFixedSize(400, 350)
        login_layout = QVBoxLayout()
        login_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        login_layout.setContentsMargins(30, 30, 30, 30)
        login_layout.setSpacing(15)

        header = QLabel("Logowanie do systemu")
        header.setObjectName("header")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Nazwa użytkownika (np. pacjent1)")
        self.pass_input = QLineEdit()
        self.pass_input.setPlaceholderText("Hasło (np. 123)")
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)

        login_btn = QPushButton("Zaloguj")
        login_btn.clicked.connect(self.handle_login)

        login_layout.addWidget(header)
        login_layout.addWidget(self.user_input)
        login_layout.addWidget(self.pass_input)
        login_layout.addSpacing(10)
        login_layout.addWidget(login_btn)

        login_container.setLayout(login_layout)

        main_layout.addWidget(login_container, alignment=Qt.AlignmentFlag.AlignCenter)
        widget.setLayout(main_layout)
        self.stacked_widget.addWidget(widget)

    def init_patient_screen(self):
        self.patient_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header and Info
        header_layout = QHBoxLayout()
        title = QLabel("Panel Pacjenta")
        title.setObjectName("header")
        self.info_label = QLabel("Oczekiwanie na wybór testu...")
        self.info_label.setStyleSheet("font-size: 16px; color: #7f8c8d;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.info_label)

        # Video Area (Scalable)
        self.video_label = QLabel("Kamera wyłączona")
        self.video_label.setObjectName("video_feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(320, 240)

        # Bottom Control Panel
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)

        self.test_selector = QComboBox()
        self.test_selector.addItems([
            "Uniesienie prawej ręki",
            "Uniesienie lewej ręki",
            "Uniesienie obu rąk",
            "Złączenie dłoni przed sobą (Palec-do-palca)"
        ])
        self.test_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.start_test_btn = QPushButton("Rozpocznij Test")
        self.start_test_btn.clicked.connect(self.start_patient_test)

        logout_btn = QPushButton("Wyloguj")
        logout_btn.setObjectName("danger")
        logout_btn.clicked.connect(self.logout)

        control_layout.addWidget(QLabel("Wybierz test:"))
        control_layout.addWidget(self.test_selector)
        control_layout.addWidget(self.start_test_btn)
        control_layout.addStretch()
        control_layout.addWidget(logout_btn)

        # Add everything to main layout
        layout.addLayout(header_layout)
        layout.addWidget(self.video_label, stretch=1)
        layout.addLayout(control_layout)

        self.patient_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.patient_widget)

    def init_doctor_screen(self):
        self.doctor_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Panel Lekarza")
        title.setObjectName("header")

        # Responsive Data Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["ID", "Pacjent", "Wynik AI", "Status"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)

        # Buttons
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Odśwież wyniki")
        refresh_btn.clicked.connect(self.load_doctor_results)

        logout_btn = QPushButton("Wyloguj")
        logout_btn.setObjectName("danger")
        logout_btn.clicked.connect(self.logout)

        btn_layout.addStretch()
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(logout_btn)

        layout.addWidget(title)
        layout.addWidget(QLabel("Ostatnie wyniki testów do przeanalizowania:"))
        layout.addWidget(self.results_table)
        layout.addLayout(btn_layout)

        self.doctor_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.doctor_widget)

    def handle_login(self):
        username = self.user_input.text()
        password = self.pass_input.text()

        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
            result = cursor.fetchone()

        if result:
            role = result[0]
            self.current_user = username
            if role == 'pacjent':
                self.stacked_widget.setCurrentIndex(1)
            elif role == 'lekarz':
                self.load_doctor_results()
                self.stacked_widget.setCurrentIndex(2)
        else:
            QMessageBox.warning(self, "Błąd", "Nieprawidłowe dane logowania!")

    def start_patient_test(self):
        self.start_test_btn.setEnabled(False)
        self.test_selector.setEnabled(False)

        test_index = self.test_selector.currentIndex()
        if test_index == 0:
            test_type = 'right_arm'
            instruction = "Proszę podnieść prawą rękę do góry."
        elif test_index == 1:
            test_type = 'left_arm'
            instruction = "Proszę podnieść lewą rękę do góry."
        elif test_index == 2:
            test_type = 'both_arms'
            instruction = "Proszę podnieść obie ręce do góry."
        elif test_index == 3:
            test_type = 'hands_together'
            instruction = "Proszę wyciągnąć ręce i złączyć dłonie przed sobą."

        self.info_label.setText(f"Test w toku: {instruction}")
        self.info_label.setStyleSheet("color: #e67e22; font-weight: bold;")

        self.tts.say(f"Rozpoczynamy badanie. {instruction}")

        self.camera_thread = CameraMediaPipeThread(test_type=test_type)
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.test_result_signal.connect(self.handle_test_success)
        self.camera_thread.start()

    def handle_test_success(self, message):
        self.info_label.setText(message)
        self.info_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        self.tts.say("Zadanie wykonane poprawnie. Dziękuję.")

        test_name = self.test_selector.currentText()
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO tests (patient_username, result_data, doctor_decision) VALUES (?, ?, ?)",
                           (self.current_user, f"Zaliczony: {test_name}", "Do weryfikacji"))

    def update_image(self, q_img):
        # Smooth scaling that dynamically matches the QLabel's current size
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def load_doctor_results(self):
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, patient_username, result_data, doctor_decision FROM tests")
            rows = cursor.fetchall()

        self.results_table.setRowCount(0)
        for row_idx, row_data in enumerate(rows):
            self.results_table.insertRow(row_idx)
            for col_idx, data in enumerate(row_data):
                item = QTableWidgetItem(str(data))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row_idx, col_idx, item)

    def logout(self):
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()

        if hasattr(self, 'start_test_btn'):
            self.start_test_btn.setEnabled(True)
            self.test_selector.setEnabled(True)
            self.info_label.setText("Oczekiwanie na wybór testu...")
            self.info_label.setStyleSheet("font-size: 16px; color: #7f8c8d;")
            self.video_label.clear()
            self.video_label.setText("Kamera wyłączona")

        self.current_user = None
        self.user_input.clear()
        self.pass_input.clear()
        self.stacked_widget.setCurrentIndex(0)

    def closeEvent(self, event):
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()
        event.accept()


if __name__ == '__main__':
    init_db()
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec())