import sys
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget


class WorkerThread(QThread):
    update_signal = Signal(int)

    def run(self):
        for i in range(10):
            self.sleep(1)  # Simulate a time-consuming task
            self.update_signal.emit(i + 1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QThread Example")

        self.label = QLabel("Waiting for update...")
        self.button = QPushButton("Start Task")
        self.button.clicked.connect(self.start_task)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.thread = WorkerThread()
        self.thread.update_signal.connect(self.update_label)

    def start_task(self):
        if not self.thread.isRunning():
            self.thread.start()

    def update_label(self, value):
        self.label.setText(f"Updated: {value}")

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
