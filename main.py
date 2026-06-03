import sys
import cv2
import sqlite3
import numpy as np
import mediapipe as mp
import pyttsx3
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QLineEdit, QStackedWidget, QMessageBox, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtTextToSpeech import QTextToSpeech
from PyQt6.QtCore import QLocale

DB_NAME = 'telemedycyna.db'


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

                            # Check if key landmarks are visible enough
                            r_visible = r_shoulder.visibility > 0.5 and r_wrist.visibility > 0.5
                            l_visible = l_shoulder.visibility > 0.5 and l_wrist.visibility > 0.5

                            passed = False

                            if self.test_type == 'right_arm' and r_visible:
                                if r_wrist.y < r_shoulder.y:
                                    passed = True

                            elif self.test_type == 'left_arm' and l_visible:
                                if l_wrist.y < l_shoulder.y:
                                    passed = True

                            elif self.test_type == 'both_arms' and r_visible and l_visible:
                                if r_wrist.y < r_shoulder.y and l_wrist.y < l_shoulder.y:
                                    passed = True

                            elif self.test_type == 'hands_together' and r_visible and l_visible:
                                # Calculate distance between wrists
                                dist = math.hypot(r_wrist.x - l_wrist.x, r_wrist.y - l_wrist.y)
                                if dist < 0.05:  # Wrists are very close to each other
                                    passed = True

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
        self.setGeometry(100, 100, 800, 700)
        self.current_user = None

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.init_login_screen()
        self.init_patient_screen()
        self.init_doctor_screen()

        self.stacked_widget.setCurrentIndex(0)

    def init_login_screen(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Nazwa użytkownika (np. pacjent1 lub lekarz1)")
        self.pass_input = QLineEdit()
        self.pass_input.setPlaceholderText("Hasło (np. 123)")
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)

        login_btn = QPushButton("Zaloguj")
        login_btn.clicked.connect(self.handle_login)

        layout.addWidget(QLabel("<h1>Logowanie do systemu</h1>"))
        layout.addWidget(self.user_input)
        layout.addWidget(self.pass_input)
        layout.addWidget(login_btn)
        widget.setLayout(layout)
        self.stacked_widget.addWidget(widget)

    def init_patient_screen(self):
        self.patient_widget = QWidget()
        layout = QVBoxLayout()

        self.info_label = QLabel("Panel Pacjenta - Oczekiwanie na test...")

        self.video_label = QLabel("Kamera wyłączona")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setMinimumSize(640, 480)

        # Dropdown for selecting the neurological test
        self.test_selector = QComboBox()
        self.test_selector.addItems([
            "Uniesienie prawej ręki",
            "Uniesienie lewej ręki",
            "Uniesienie obu rąk",
            "Złączenie dłoni przed sobą (Palec-do-palca)"
        ])

        self.start_test_btn = QPushButton("Rozpocznij Test Neurologiczny")
        self.start_test_btn.clicked.connect(self.start_patient_test)

        logout_btn = QPushButton("Wyloguj")
        logout_btn.clicked.connect(self.logout)

        layout.addWidget(self.info_label)
        layout.addWidget(self.video_label)
        layout.addWidget(QLabel("Wybierz rodzaj testu do przeprowadzenia:"))
        layout.addWidget(self.test_selector)
        layout.addWidget(self.start_test_btn)
        layout.addWidget(logout_btn)
        self.patient_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.patient_widget)

    def init_doctor_screen(self):
        self.doctor_widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("<h1>Panel Lekarza</h1>"))
        layout.addWidget(QLabel("Ostatnie wyniki testów do przeanalizowania (Z bazy danych):"))

        self.results_label = QLabel("Brak nowych testów.")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        refresh_btn = QPushButton("Odśwież wyniki")
        refresh_btn.clicked.connect(self.load_doctor_results)

        logout_btn = QPushButton("Wyloguj")
        logout_btn.clicked.connect(self.logout)

        layout.addWidget(self.results_label)
        layout.addWidget(refresh_btn)
        layout.addWidget(logout_btn)
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

        self.info_label.setText(f"Test w toku... {instruction}")
        self.info_label.setStyleSheet("color: black; font-weight: normal;")

        self.tts.say(f"Rozpoczynamy badanie. {instruction}")

        self.camera_thread = CameraMediaPipeThread(test_type=test_type)
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.test_result_signal.connect(self.handle_test_success)
        self.camera_thread.start()

    def handle_test_success(self, message):
        self.info_label.setText(message)
        self.info_label.setStyleSheet("color: green; font-weight: bold;")
        self.tts.say("Zadanie wykonane poprawnie. Dziękuję.")

        # Save the completed test to the database
        test_name = self.test_selector.currentText()
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO tests (patient_username, result_data, doctor_decision) VALUES (?, ?, ?)",
                           (self.current_user, f"Zaliczony: {test_name}", "Do weryfikacji"))

    def update_image(self, q_img):
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def load_doctor_results(self):
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, patient_username, result_data, doctor_decision FROM tests")
            rows = cursor.fetchall()

        if rows:
            txt = "\n\n".join([f"ID: {r[0]} | Pacjent: {r[1]}\nWynik AI: {r[2]} | Status: {r[3]}" for r in rows])
            self.results_label.setText(txt)
        else:
            self.results_label.setText("Brak danych w bazie.")

    def logout(self):
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()

        if hasattr(self, 'start_test_btn'):
            self.start_test_btn.setEnabled(True)
            self.test_selector.setEnabled(True)
            self.info_label.setText("Panel Pacjenta - Oczekiwanie na test...")
            self.info_label.setStyleSheet("color: black; font-weight: normal;")
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