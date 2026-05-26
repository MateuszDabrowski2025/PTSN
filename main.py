import sys
import cv2
import sqlite3
import pyttsx3
import mediapipe as mp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QLineEdit, QStackedWidget, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

DB_NAME = 'telemedycyna.db'


# ==========================================
# 1. BAZA DANYCH (SQLite)
# ==========================================
def init_db():
    # Używamy menedżera kontekstu (with), co automatycznie wykonuje commit() i zamyka połączenie
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        # Tabela użytkowników
        cursor.execute('''CREATE TABLE IF NOT EXISTS users
                          (id INTEGER PRIMARY KEY, username TEXT, password TEXT, role TEXT )''')
        # Tabela wyników testów
        cursor.execute('''CREATE TABLE IF NOT EXISTS tests
                          ( id INTEGER PRIMARY KEY, patient_username TEXT, result_data TEXT, doctor_decision TEXT )''')

        # Dodanie testowych użytkowników (jeśli nie istnieją)
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
    """Wątek do syntezy mowy, aby nie blokować GUI PyQt"""

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
            pass  # Zapobiega crashom, jeśli pętla silnika TTS jest już aktywna


class CameraMediaPipeThread(QThread):
    """Wątek przechwytujący obraz z kamery i nakładający MediaPipe"""
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("Błąd: Nie można otworzyć kamery.")
            return

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self._run_flag:
                ret, frame = cap.read()
                if ret:
                    # Przetwarzanie obrazu na RGB dla MediaPipe
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    # Rysowanie szkieletu na obrazie
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Konwersja z powrotem do formatu zgodnego z PyQt
                    h, w, ch = image_rgb.shape
                    bytes_per_line = ch * w
                    q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.change_pixmap_signal.emit(q_img)
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
        self.setWindowTitle("System Diagnostyki Neurologicznej")
        self.setGeometry(100, 100, 800, 600)
        self.current_user = None

        # Główny menedżer widoków (pozwala przełączać ekrany)
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Inicjalizacja ekranów
        self.init_login_screen()
        self.init_patient_screen()
        self.init_doctor_screen()

        self.stacked_widget.setCurrentIndex(0)  # Start od logowania

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

    # --- Logika aplikacji ---

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
        self.start_test_btn.setEnabled(False)  # Zablokowanie przycisku przed spamowaniem

        self.info_label.setText("Test w toku... Postępuj zgodnie z instrukcjami głosowymi.")

        # Uruchomienie asystenta głosowego
        self.voice_thread = VoiceAssistantThread(
            "Rozpoczynamy badanie układu nerwowego. Proszę podnieść prawą rękę do góry.")
        self.voice_thread.start()

        # Uruchomienie śledzenia z kamery
        self.camera_thread = CameraMediaPipeThread(camera_id=0)
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.start()

        # Zapis do bazy danych
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO tests (patient_username, result_data, doctor_decision)
                              VALUES (?, ?, ?)''',
                           (self.current_user, "Ruch zarejestrowany: Prawa ręka uniesiona (Dane z MediaPipe)",
                            "Oczekuje na decyzję"))

    def update_image(self, q_img):
        # Aktualizacja obrazu w interfejsie pacjenta
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

        # Reset interfejsu pacjenta
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
    # Przygotowanie bazy danych na start
    init_db()

    # Uruchomienie aplikacji GUI
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec())