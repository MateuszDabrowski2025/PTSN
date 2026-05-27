import sys
import cv2
import sqlite3
import numpy as np
import mediapipe as mp
import pyttsx3
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QLineEdit, QStackedWidget, QMessageBox)
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
                          (id INTEGER PRIMARY KEY, username TEXT, password TEXT, role TEXT )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS tests
                          ( id INTEGER PRIMARY KEY, patient_username TEXT, result_data TEXT, doctor_decision TEXT )''')

        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                           ('pacjent1', '123', 'pacjent'))
            cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                           ('lekarz1', '123', 'lekarz'))


# ==========================================
# 2. WĄTKI POBOCZNE (Audio i Wideo)
# ==========================================
class VoiceAssistantThread(QThread):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'polish' in voice.name.lower() or 'pl' in voice.languages:
                engine.setProperty('voice', voice.id)
                break
        try:
            engine.say(self.text)
            engine.runAndWait()
        except RuntimeError:
            pass

class CameraMediaPipeThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    # Now we can send the test result back to the main GUI!
    test_result_signal = pyqtSignal(str)

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Błąd: Nie można otworzyć kamery o ID {self.camera_id}.")
            return
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        # mp_pose and mp_drawing are now safely pulled from the global scope
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
                        # In MediaPipe, Y=0 is the top of the screen and Y=1 is the bottom.
                        # We extract the Right Shoulder (12) and Right Wrist (16)
                        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                        # If the wrist's Y coordinate is smaller than the shoulder's Y, the arm is raised
                        if right_wrist.y < right_shoulder.y:
                            self.test_result_signal.emit("Sukces: Prawa ręka została uniesiona.")

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
        self.setGeometry(100, 100, 800, 600)
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

        self.start_test_btn = QPushButton("Rozpocznij Test Neurologiczny")
        self.start_test_btn.clicked.connect(self.start_patient_test)

        logout_btn = QPushButton("Wyloguj")
        logout_btn.clicked.connect(self.logout)

        layout.addWidget(self.info_label)
        layout.addWidget(self.video_label)
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
        self.info_label.setText("Test w toku... Proszę podnieść prawą rękę do góry.")
        self.tts.say("Rozpoczynamy badanie układu nerwowego. Proszę podnieść prawą rękę do góry.")

        self.camera_thread = CameraMediaPipeThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        # Connect the physical detection logic!
        self.camera_thread.test_result_signal.connect(self.handle_test_success)
        self.camera_thread.start()

    def handle_test_success(self, message):
        # Update the screen
        self.info_label.setText(message)
        self.info_label.setStyleSheet("color: green; font-weight: bold;")

    def update_image(self, q_img):
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def load_doctor_results(self):
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, patient_username, result_data FROM tests")
            rows = cursor.fetchall()

        if rows:
            txt = "\n".join([f"ID: {r[0]} | Pacjent: {r[1]} | Wynik AI: {r[2]}" for r in rows])
            self.results_label.setText(txt)
        else:
            self.results_label.setText("Brak danych w bazie.")

    def logout(self):
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()

        if hasattr(self, 'start_test_btn'):
            self.start_test_btn.setEnabled(True)
            self.info_label.setText("Panel Pacjenta - Oczekiwanie na test...")
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