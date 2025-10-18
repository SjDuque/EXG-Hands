from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from src.exg.graph_client import GraphClient
from src.exg.brainflow_server import BrainFlowServer
from src.exg.serial_server import CircuitPySerialServer
from hand_tracking.prompt_client import HandWidget
from hand_tracking.prompt_server import PromptServer
from src.csv.record_lsl import RecorderWorker, SESSION_DIR
from src.ai.model_server import ModelServer


@dataclass
class FilterSettings:
    lowpass: Optional[float]
    highpass: Optional[float]
    notch: Optional[list[float]]


class ParameterPanel(QtWidgets.QGroupBox):
    """Collects runtime parameters for servers and recording."""

    def __init__(
        self,
        default_session: str,
        default_notch: list[float] = [50.0, 60.0],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__("Run Parameters", parent)
        self.setObjectName("parameter-panel")
        self.setMinimumHeight(160)

        self._default_session = default_session
        self.default_notch = default_notch

        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.session_input = QtWidgets.QLineEdit()
        self.session_input.setPlaceholderText("s_MM_DD_YY or data/s_MM_DD_YY")
        self.session_input.setText(default_session)
        layout.addRow("Session name", self.session_input)

        self.highpass_input = QtWidgets.QLineEdit()
        self.highpass_input.setPlaceholderText("High-pass Hz (e.g. 10)")
        layout.addRow("High-pass", self.highpass_input)

        self.lowpass_input = QtWidgets.QLineEdit()
        self.lowpass_input.setPlaceholderText("Low-pass Hz (e.g. half sampling rate)")
        layout.addRow("Low-pass", self.lowpass_input)

        self.notch_input = QtWidgets.QLineEdit()
        self.notch_input.setPlaceholderText("Notch Hz list (e.g. 50,60)")
        self.notch_input.setText(
            ", ".join(self._format_float(value) for value in self.default_notch)
        )
        layout.addRow("Notch", self.notch_input)

        # Extra parameters (optional free-text field)
        # self.extra_input = QtWidgets.QLineEdit()
        # self.extra_input.setPlaceholderText("Extra parameters (optional)")
        # layout.addRow("Extra", self.extra_input)

    @staticmethod
    def _format_float(value: float) -> str:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"

    def _parse_float(self, text: str) -> float | None:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value: '{text}'.") from exc
        if value <= 0:
            raise ValueError("Frequencies must be > 0.")
        return value

    def collect_filter_settings(self) -> FilterSettings:
        highpass = self._parse_float(self.highpass_input.text())
        lowpass = self._parse_float(self.lowpass_input.text())

        notch_values: list[float] = []
        notch_text = self.notch_input.text().strip()
        if notch_text:
            for part in notch_text.split(","):
                freq = self._parse_float(part)
                if freq is not None:
                    notch_values.append(freq)

        if not notch_values:
            notch_values = self.default_notch

        return FilterSettings(lowpass=lowpass, highpass=highpass, notch=notch_values)

    def resolve_session_directory(self) -> str:
        text = self.session_input.text().strip()
        if not text:
            raise ValueError("Session name cannot be empty.")
        if os.path.isabs(text):
            return text
        if text.startswith("data/"):
            return text
        return os.path.join("data", text)

    def extra_parameters(self) -> str:
        return self.extra_input.text().strip()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EXG & Hand Control")
        self.resize(1280, 800)

        self.brainflow_server: Optional[BrainFlowServer] = None
        self.serial_server: Optional[CircuitPySerialServer] = None
        self.prompt_server: Optional[PromptServer] = None
        self.recorder_worker: Optional[RecorderWorker] = None

        self.graph_widget: Optional[GraphClient] = None
        # Primary hand widget shows prompt server
        self.hand_widget = HandWidget(flip=True, auto_start=False)
        # Secondary hand widget shows model output (inference)
        self.model_hand_widget = HandWidget(flip=False, stream_name="model_output", auto_start=False)
        self.parameter_panel: Optional[ParameterPanel] = None
        self._session_root = os.path.dirname(SESSION_DIR) or "data"
        self.model_server: Optional[ModelServer] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _default_session_name(self) -> str:
        return f"s_{datetime.now().strftime('%m_%d_%y')}"

    def _default_session_directory(self) -> str:
        return os.path.join(self._session_root, self._default_session_name())

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.graph_stack = QtWidgets.QStackedWidget()
        placeholder = QtWidgets.QLabel("Start a data stream to view EXG traces.")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.graph_stack.addWidget(placeholder)
        splitter.addWidget(self.graph_stack)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_top = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_top)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        buttons_layout = QtWidgets.QGridLayout()
        buttons_layout.setHorizontalSpacing(8)
        buttons_layout.setVerticalSpacing(8)

        self.brainflow_btn = QtWidgets.QPushButton("Start BrainFlow Server")
        self.brainflow_btn.setCheckable(True)
        self.brainflow_btn.toggled.connect(self._toggle_brainflow)
        buttons_layout.addWidget(self.brainflow_btn, 0, 0)

        self.serial_btn = QtWidgets.QPushButton("Start Serial Server")
        self.serial_btn.setCheckable(True)
        self.serial_btn.toggled.connect(self._toggle_serial)
        buttons_layout.addWidget(self.serial_btn, 0, 1)

        self.prompt_btn = QtWidgets.QPushButton("Start Prompt Server")
        self.prompt_btn.setCheckable(True)
        self.prompt_btn.toggled.connect(self._toggle_prompt)
        buttons_layout.addWidget(self.prompt_btn, 1, 0)

        # Model controls
        self.model_btn = QtWidgets.QPushButton("Start Model Server")
        self.model_btn.setCheckable(True)
        self.model_btn.toggled.connect(self._toggle_model_server)
        buttons_layout.addWidget(self.model_btn, 0, 2)

        # Save model button (inactive until model server exists)
        self.save_model_btn = QtWidgets.QPushButton("Save Model")
        self.save_model_btn.setEnabled(False)
        self.save_model_btn.clicked.connect(self._on_save_model)
        buttons_layout.addWidget(self.save_model_btn, 0, 3)

        self.mode_btn = QtWidgets.QPushButton("Mode: Train")
        self.mode_btn.setCheckable(True)
        self.mode_btn.toggled.connect(self._toggle_model_mode)
        buttons_layout.addWidget(self.mode_btn, 1, 2)

        # Model metrics display
        self.loss_label = QtWidgets.QLabel("Loss: -")
        self.loss_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        buttons_layout.addWidget(self.loss_label, 2, 2)

        self.r2_label = QtWidgets.QLabel("R2: -")
        self.r2_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        buttons_layout.addWidget(self.r2_label, 2, 3)

        self.recorder_btn = QtWidgets.QPushButton("Start Recorder")
        self.recorder_btn.setCheckable(True)
        self.recorder_btn.toggled.connect(self._toggle_recorder)
        buttons_layout.addWidget(self.recorder_btn, 1, 1)

        right_layout.addLayout(buttons_layout)

        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        right_layout.addWidget(self.status_label)

        # Place the prompt and model hand widgets side-by-side
        hands_container = QtWidgets.QWidget()
        hands_layout = QtWidgets.QHBoxLayout(hands_container)
        hands_layout.setContentsMargins(0, 0, 0, 0)
        hands_layout.setSpacing(8)

        # Left column: Prompt window (header + checkbox + widget)
        prompt_col = QtWidgets.QWidget()
        prompt_col_layout = QtWidgets.QVBoxLayout(prompt_col)
        prompt_col_layout.setContentsMargins(0, 0, 0, 0)
        prompt_col_layout.setSpacing(4)

        prompt_header = QtWidgets.QHBoxLayout()
        prompt_label = QtWidgets.QLabel("Prompt Window")
        prompt_label.setStyleSheet("font-weight: bold")
        prompt_header.addWidget(prompt_label)
        self.prompt_enable = QtWidgets.QCheckBox("Enabled")
        self.prompt_enable.setChecked(True)
        self.prompt_enable.toggled.connect(lambda v: self._set_widget_enabled(self.hand_widget, v))
        prompt_header.addStretch()
        prompt_header.addWidget(self.prompt_enable)
        prompt_col_layout.addLayout(prompt_header)
        prompt_col_layout.addWidget(self.hand_widget, stretch=1)

        # Right column: Model output window
        model_col = QtWidgets.QWidget()
        model_col_layout = QtWidgets.QVBoxLayout(model_col)
        model_col_layout.setContentsMargins(0, 0, 0, 0)
        model_col_layout.setSpacing(4)

        model_header = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("Model Output Window")
        model_label.setStyleSheet("font-weight: bold")
        model_header.addWidget(model_label)
        self.model_enable = QtWidgets.QCheckBox("Enabled")
        self.model_enable.setChecked(True)
        self.model_enable.toggled.connect(lambda v: self._set_widget_enabled(self.model_hand_widget, v))
        model_header.addStretch()
        model_header.addWidget(self.model_enable)
        model_col_layout.addLayout(model_header)
        model_col_layout.addWidget(self.model_hand_widget, stretch=1)

        hands_layout.addWidget(prompt_col, stretch=1)
        hands_layout.addWidget(model_col, stretch=1)

        right_layout.addWidget(hands_container, stretch=1)

        right_splitter.addWidget(right_top)

        self.parameter_panel = ParameterPanel(
            default_session=self._default_session_name(),
            parent=right_splitter,
        )
        right_splitter.addWidget(self.parameter_panel)
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)

        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _toggle_brainflow(self, checked: bool) -> None:
        if checked:
            try:
                filters = (
                    self.parameter_panel.collect_filter_settings()
                    if self.parameter_panel is not None
                    else FilterSettings(lowpass=None, highpass=10.0, notch=[50.0, 60.0])
                )
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid Filter Parameters", str(exc))
                self.brainflow_btn.setChecked(False)
                return

            try:
                server = BrainFlowServer(
                    # board_id=BOARD_ID,
                    is_bipolar=True,
                    lowpass_fs=filters.lowpass,
                    highpass_fs=filters.highpass,
                    notch_fs_list=filters.notch,
                )
                lowpass = server.lowpass
                highpass = server.highpass
                notches = server.notch_list
                server.daemon = True
                server.start()
                self.brainflow_server = server
                self.brainflow_btn.setText("Stop BrainFlow")
                description = (
                    f"BrainFlow running — HP {highpass} Hz, "
                    f"LP {lowpass} Hz"
                    f", Notch {', '.join(str(n) for n in notches)} Hz"
                )
                self._set_status(description)
                QtCore.QTimer.singleShot(1000, self._ensure_graph_widget)
            except Exception as exc:
                logging.exception("Failed to start BrainFlow Server")
                QtWidgets.QMessageBox.critical(self, "BrainFlow Error", str(exc))
                self.brainflow_btn.setChecked(False)
        else:
            self._stop_brainflow()

    def _toggle_serial(self, checked: bool) -> None:
        if checked:
            try:
                server = CircuitPySerialServer()
                server.start()
                self.serial_server = server
                self.serial_btn.setText("Stop Serial Server")
                self._set_status("Serial server streaming")
                QtCore.QTimer.singleShot(1000, self._ensure_graph_widget)
            except Exception as exc:
                logging.exception("Failed to start serial server")
                QtWidgets.QMessageBox.critical(self, "Serial Server Error", str(exc))
                self.serial_btn.setChecked(False)
        else:
            self._stop_serial()

    def _toggle_prompt(self, checked: bool) -> None:
        if checked:
            try:
                streamer = PromptServer()
                streamer.daemon = True
                streamer.start()
                self.prompt_server = streamer
                self.prompt_btn.setText("Stop Prompt Server")
                self._set_status("Prompt server running")
                # Start the prompt hand widget stream
                self.hand_widget.start_stream()
            except Exception as exc:
                logging.exception("Failed to start prompt server")
                QtWidgets.QMessageBox.critical(self, "Prompt Server Error", str(exc))
                self.prompt_btn.setChecked(False)
        else:
            self._stop_prompt()

    def _toggle_recorder(self, checked: bool) -> None:
        if checked:
            try:
                session_dir = (
                    self.parameter_panel.resolve_session_directory()
                    if self.parameter_panel is not None
                    else self._default_session_directory()
                )
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid Session", str(exc))
                self.recorder_btn.setChecked(False)
                return

            try:
                worker = RecorderWorker(
                    streams={"exg": "raw_exg", "prompt": "finger_prompt"},
                    session_dir=session_dir,
                )
                worker.status_changed.connect(self._set_status)
                worker.start()
                self.recorder_worker = worker
                self.recorder_btn.setText("Stop Recorder")
            except Exception as exc:
                logging.exception("Failed to start recorder")
                QtWidgets.QMessageBox.critical(self, "Recorder Error", str(exc))
                self.recorder_btn.setChecked(False)
        else:
            self._stop_recorder()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_graph_widget(self) -> None:
        if self.graph_widget is not None:
            return

        try:
            widget = GraphClient(stream_name="filtered_exg")
        except RuntimeError:
            try:
                widget = GraphClient(stream_name="raw_exg")
            except RuntimeError:
                QtCore.QTimer.singleShot(1000, self._ensure_graph_widget)
                return

        self.graph_widget = widget
        self.graph_stack.addWidget(widget)
        self.graph_stack.setCurrentWidget(widget)

    def _stop_brainflow(self) -> None:
        if self.brainflow_server:
            self.brainflow_server.stop()
            try:
                # BrainFlowServer is a threading.Thread or similar; prefer wait if available
                self.brainflow_server.wait(2000)
            except Exception:
                try:
                    self.brainflow_server.join(timeout=2.0)
                except Exception:
                    pass
            self.brainflow_server = None
        self.brainflow_btn.setText("Start BrainFlow")
        self._set_status("BrainFlow server stopped")

    def _stop_serial(self) -> None:
        if self.serial_server:
            self.serial_server.stop()
            try:
                self.serial_server.wait(2000)
            except Exception:
                try:
                    self.serial_server.join(timeout=2.0)
                except Exception:
                    pass
            self.serial_server = None
        self.serial_btn.setText("Start Serial Server")
        self._set_status("Serial server stopped")

    def _stop_prompt(self) -> None:
        if self.prompt_server:
            self.prompt_server.stop()
            try:
                self.prompt_server.wait(2000)
            except Exception:
                try:
                    self.prompt_server.join(timeout=2.0)
                except Exception:
                    pass
            self.prompt_server = None
        if hasattr(self.hand_widget, "stop_stream"):
            self.hand_widget.stop_stream()
        self.prompt_btn.setText("Start Prompt Server")
        self._set_status("Prompt server stopped")

    def _toggle_model_server(self, checked: bool) -> None:
        if checked:
            try:
                server = ModelServer()
                server.daemon = True
                server.start()
                self.model_server = server
                # connect metrics signal
                try:
                    server.metrics_updated.connect(self._on_model_metrics)
                except Exception:
                    pass
                self.model_btn.setText("Stop Model Server")
                self.save_model_btn.setEnabled(True)
                self._set_status("Model server running (train mode)")
                # Start the model hand widget to read model output (it will resolve stream)
                self.model_hand_widget.start_stream()
            except Exception as exc:
                logging.exception("Failed to start model server")
                QtWidgets.QMessageBox.critical(self, "Model Server Error", str(exc))
                self.model_btn.setChecked(False)
        else:
            if self.model_server:
                try:
                    self.model_server.stop()
                    try:
                        self.model_server.wait(2000)
                    except Exception:
                        self.model_server.join(timeout=2.0)
                except Exception:
                    pass
                # disconnect metrics and clear labels
                try:
                    self.model_server.metrics_updated.disconnect(self._on_model_metrics)
                except Exception:
                    pass
                self.loss_label.setText("Loss: -")
                self.r2_label.setText("R2: -")
                self.model_server = None
            self.save_model_btn.setEnabled(False)
            self.model_btn.setText("Start Model Server")
            self._set_status("Model server stopped")

    def _toggle_model_mode(self, checked: bool) -> None:
        # checked==True -> inference mode
        if not self.model_server:
            # toggle button should reflect mode only when server running
            self.mode_btn.setChecked(False)
            return
        try:
            mode = "inference" if checked else "train"
            self.model_server.set_mode(mode)
            self.mode_btn.setText(f"Mode: {mode.capitalize()}")
            self._set_status(f"Model server mode: {mode}")
            # If switching to inference, ensure model output widget is enabled and streaming
            if mode == "inference":
                self.model_hand_widget.start_stream()
            else:
                # in train mode, the model output may not be produced; but keep widget available
                pass
        except Exception:
            pass

    def _on_model_metrics(self, loss: float, r2: float) -> None:
        try:
            self.loss_label.setText(f"Loss: {loss:.4f}")
            self.r2_label.setText(f"R2: {r2:.4f}")
        except Exception:
            pass

    def _stop_recorder(self) -> None:
        if self.recorder_worker:
            self.recorder_worker.stop()
            self.recorder_worker = None
        self.recorder_btn.setText("Start Recorder")

    def _set_widget_enabled(self, widget: QtWidgets.QWidget, enabled: bool) -> None:
        widget.setVisible(enabled)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(f"Status: {message}")

    def _on_save_model(self) -> None:
        if not self.model_server:
            QtWidgets.QMessageBox.warning(self, "No Model", "Model server is not running.")
            return
        # Default path: <session_dir>/models/model_<timestamp>.keras
        try:
            session_dir = (
                self.parameter_panel.resolve_session_directory()
                if self.parameter_panel is not None
                else self._default_session_directory()
            )
        except Exception:
            session_dir = self._default_session_directory()
        default_dir = os.path.join(session_dir, "models")
        try:
            os.makedirs(default_dir, exist_ok=True)
        except Exception:
            default_dir = "."
        path = os.path.join(default_dir, f"model_{int(time.time())}.keras")
        try:
            # Request save on the model server thread
            self.model_server.request_save(path)
            QtWidgets.QMessageBox.information(self, "Save Requested", f"Model save requested: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Failed", str(exc))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        self._stop_recorder()
        self._stop_prompt()
        self._stop_serial()
        self._stop_brainflow()

        # Stop model server and model hand widget
        try:
            if self.model_server:
                self.model_server.stop()
                try:
                    self.model_server.wait(1000)
                except Exception:
                    try:
                        self.model_server.join(timeout=1.0)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if hasattr(self.model_hand_widget, "shutdown"):
                self.model_hand_widget.shutdown()
        except Exception:
            pass

        if self.graph_widget:
            self.graph_widget.shutdown()
            self.graph_widget = None

        if hasattr(self.hand_widget, "shutdown"):
            self.hand_widget.shutdown()

        for button in (
            self.brainflow_btn,
            self.serial_btn,
            self.prompt_btn,
            self.recorder_btn,
        ):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)

        super().closeEvent(event)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
