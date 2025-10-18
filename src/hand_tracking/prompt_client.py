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

    def __init__(self, stream_name: str = "finger_prompt", smoothing: float = 0.2, retry_interval: float = 0.5):
        super().__init__()
        self.stream_name = stream_name
        self.smoothing = smoothing
        self.retry_interval = max(retry_interval, 0.1)
        self.running = True
        # start with a reasonable default of 5 fingers, but allow dynamic resizing
        self.target_pose = [0.0] * 5
        self.current_pose = [0.0] * 5
        self._inlet: StreamInlet | None = None

    def run(self):
        # Resolve inlet (retry loop)
        while self.running and self._inlet is None:
            try:
                streams = resolve_byprop('name', self.stream_name, timeout=self.retry_interval)
            except Exception:
                streams = []
            if streams:
                try:
                    self._inlet = StreamInlet(streams[0])
                except Exception:
                    self._inlet = None
                    self.msleep(int(self.retry_interval * 1000))
            else:
                self.msleep(int(self.retry_interval * 1000))

        if not self._inlet:
            return

        while self.running:
            try:
                sample, _ = self._inlet.pull_sample(timeout=0.0)
            except Exception:
                sample = None

            if sample:
                # Normalize incoming values and coerce to 5-length mapping.
                incoming = [float(x) for x in sample]
                # If upstream sends 4 values (thumb,index,middle,ring_or_pinky), map to 5 by
                # duplicating the 4th value to both ring and pinky positions.
                if len(incoming) == 4:
                    new_target = [1.0 - incoming[0], 1.0 - incoming[1], 1.0 - incoming[2], 1.0 - incoming[3], 1.0 - incoming[3]]
                else:
                    # If incoming has >=5, take first 5 and invert; if fewer, pad with zeros
                    arr = incoming[:5] + [0.0] * max(0, 5 - len(incoming))
                    new_target = [1.0 - v for v in arr]

                # Ensure current_pose is length 5
                if len(self.current_pose) != 5:
                    new_current = [0.0] * 5
                    for i in range(min(len(self.current_pose), 5)):
                        new_current[i] = self.current_pose[i]
                    self.current_pose = new_current
                self.target_pose = new_target

            # Smooth towards 5-element target
            for i in range(5):
                self.current_pose[i] += self.smoothing * (self.target_pose[i] - self.current_pose[i])

            # Emit a copy to avoid sharing mutable list across threads
            try:
                self.pose_updated.emit(list(self.current_pose))
            except Exception:
                pass

            self.msleep(33)  # ~30 FPS

    def stop(self):
        self.running = False
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
            self._inlet = None
        self.quit()
        self.wait()

class HandWidget(QWidget):
    def __init__(self, flip: bool = False, stream_name: str = "finger_prompt", auto_start: bool = False):
        super().__init__()
        self.setFixedSize(400, 400)
        self.flip = flip
        self.pose = [0.0] * 5
        self.stream_name = stream_name

        self.worker: LSLWorker | None = None

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

        if auto_start:
            self.start_stream()

    def update_pose(self, new_pose):
        # copy the incoming pose to avoid accidental shared-mutation
        try:
            self.pose = list(new_pose)
        except Exception:
            # fallback: keep existing pose
            pass

    def start_stream(self) -> None:
        # Guard against C++ wrapper being deleted; treat as not running in that case
        try:
            if getattr(self, 'worker', None) and self.worker.isRunning():
                return
        except RuntimeError:
            # underlying C++ worker was deleted; reset reference
            try:
                self.worker = None
            except Exception:
                pass
        worker = LSLWorker(stream_name=self.stream_name)
        worker.pose_updated.connect(self.update_pose)
        worker.finished.connect(worker.deleteLater)
        self.worker = worker
        self.worker.start()

    def stop_stream(self) -> None:
        if not self.worker:
            return
        try:
            self.worker.pose_updated.disconnect(self.update_pose)
        except Exception:
            pass
        self.worker.stop()
        self.worker = None

    def shutdown(self):
        """Stop background threads and timers."""
        self.stop_stream()
        self.timer.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
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
            available_fingers = [i for i in finger_indices if i < len(self.FINGER_LENGTHS) and i < len(self.pose)]

            for pos, i in enumerate(available_fingers):
                try:
                    p = float(self.pose[i])
                except Exception:
                    p = 0.0
                # clamp p to [0,1]
                p = max(0.0, min(1.0, p))
                length = int(self.FINGER_LENGTHS[i] * (1.0 - 0.25 * p))
                base_x = int(top_left_x + pos * spacing)
                drop = int(p * length / 1.5)
                base_y = int(top_left_y + drop)
                tip_y = int(base_y - length)
                # clamp coordinates to reasonable window bounds
                base_x = max(0, min(self.width(), base_x))
                base_y = max(0, min(self.height(), base_y))
                tip_y = max(-2147483647, min(2147483647, tip_y))

                painter.setPen(QPen(self.FINGER_COLORS[i], 6))
                painter.drawLine(base_x, base_y, base_x, tip_y)

            # Thumb
            i = 0
            if len(self.pose) > 0:
                try:
                    p0 = float(self.pose[0])
                except Exception:
                    p0 = 0.0
            else:
                p0 = 0.0

            thumb_length = int(self.FINGER_LENGTHS[i] * (1.0 - 0.25 * p0))
            thumb_base_y = top_left_y + self.PALM_SIZE
            color = self.FINGER_COLORS[i]

            if self.flip:
                base = (top_left_x + self.PALM_SIZE, thumb_base_y)
                angle = math.radians(315 * (1 - p0) + 225 * p0)
                tip = (
                    int(base[0] + thumb_length * math.cos(angle)),
                    int(base[1] + thumb_length * math.sin(angle))
                )
            else:
                base = (top_left_x, thumb_base_y)
                angle = math.radians(135 * (1 - p0) + 45 * p0)
                tip = (
                    int(base[0] + thumb_length * math.cos(angle)),
                    int(base[1] - thumb_length * math.sin(angle))
                )

            painter.setPen(QPen(color, 6))
            painter.drawLine(*base, *tip)
        finally:
            try:
                painter.end()
            except Exception:
                pass

    def closeEvent(self, event):
        self.shutdown()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Allow Python to catch SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    w = HandWidget(flip=True, auto_start=True)
    w.setWindowTitle("Hand Prompt Client")
    w.show()
    
    # Optional: ensure app responsiveness to Ctrl+C by starting a QTimer
    timer = QTimer()
    timer.start(100)
    timer.timeout.connect(lambda: None)
    
    sys.exit(app.exec_())
