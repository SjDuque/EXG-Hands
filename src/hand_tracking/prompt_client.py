import sys
import math
import numpy as np
import signal
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from pylsl import StreamInlet, resolve_byprop

class LSLWorker(QThread):
    pose_updated = pyqtSignal(list)

    def __init__(self, stream_name="finger_prompt", smoothing=0.2):
        super().__init__()
        self.stream_name = stream_name
        self.smoothing = smoothing
        self.running = True
        self.target_pose = [0.0] * 5
        self.current_pose = [0.0] * 5

    def run(self):
        streams = resolve_byprop('name', self.stream_name)
        inlet = StreamInlet(streams[0])

        while self.running:
            sample, _ = inlet.pull_sample(timeout=0.0)
            if sample:
                self.target_pose = [1 - float(x) for x in sample]
                QApplication.beep()
            for i in range(5):
                self.current_pose[i] += self.smoothing * (self.target_pose[i] - self.current_pose[i])
            self.pose_updated.emit(self.current_pose)
            self.msleep(33)  # ~30 FPS

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class HandWidget(QWidget):
    def __init__(self, flip=False):
        super().__init__()
        self.setFixedSize(400, 400)
        self.flip = flip
        self.pose = [0.0] * 5

        self.worker = LSLWorker()
        self.worker.pose_updated.connect(self.update_pose)
        self.worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(33)

        # Constants
        self.PALM_SIZE = 75
        self.FINGER_LENGTHS = [60, 70, 90, 80, 60]
        self.FINGER_COLORS = [
            QColor(255, 100, 100),  # thumb
            QColor(100, 255, 100),  # index
            QColor(100, 100, 255),  # middle
            QColor(255, 255, 100),  # ring
            QColor(255, 100, 255),  # pinky
        ]

    def update_pose(self, new_pose):
        self.pose = new_pose

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 10, 10))

        center_x = self.width() // 2
        center_y = self.height() // 2
        top_left_x = center_x - self.PALM_SIZE // 2
        top_left_y = center_y - self.PALM_SIZE // 2

        # Draw palm
        painter.setPen(QPen(QColor(150, 150, 150), 2))
        painter.drawRect(top_left_x, top_left_y, self.PALM_SIZE, self.PALM_SIZE)

        # Draw fingers
        spacing = self.PALM_SIZE // 3
        finger_indices = [4, 3, 2, 1] if self.flip else [1, 2, 3, 4]

        for pos, i in enumerate(finger_indices):
            length = int(self.FINGER_LENGTHS[i] * (1.0 - 0.25 * self.pose[i]))
            base_x = top_left_x + pos * spacing
            drop = int(self.pose[i] * length / 1.5)
            base_y = top_left_y + drop
            tip_y = base_y - length

            painter.setPen(QPen(self.FINGER_COLORS[i], 6))
            painter.drawLine(base_x, base_y, base_x, tip_y)

        # Thumb
        i = 0
        thumb_length = self.FINGER_LENGTHS[i] * (1.0 - 0.25 * self.pose[i])
        thumb_base_y = top_left_y + self.PALM_SIZE
        color = self.FINGER_COLORS[i]

        if self.flip:
            base = (top_left_x + self.PALM_SIZE, thumb_base_y)
            angle = math.radians(315 * (1 - self.pose[i]) + 225 * self.pose[i])
            tip = (
                int(base[0] + thumb_length * math.cos(angle)),
                int(base[1] + thumb_length * math.sin(angle))
            )
        else:
            base = (top_left_x, thumb_base_y)
            angle = math.radians(135 * (1 - self.pose[i]) + 45 * self.pose[i])
            tip = (
                int(base[0] + thumb_length * math.cos(angle)),
                int(base[1] - thumb_length * math.sin(angle))
            )

        painter.setPen(QPen(color, 6))
        painter.drawLine(*base, *tip)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Allow Python to catch SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    w = HandWidget(flip=True)
    w.setWindowTitle("Hand Prompt Client")
    w.show()
    
    # Optional: ensure app responsiveness to Ctrl+C by starting a QTimer
    timer = QTimer()
    timer.start(100)
    timer.timeout.connect(lambda: None)
    
    sys.exit(app.exec_())
