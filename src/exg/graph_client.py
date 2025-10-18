import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from pylsl import StreamInlet, resolve_byprop
from typing import Literal

class GraphClient(QtWidgets.QWidget):
    """A PyQt widget that visualizes an LSL stream in real time using pyqtgraph."""

    def __init__(
        self,
        stream_name: str = "filtered_exg",
        window_secs: float = 5.0,
        refresh_hz: int = 60,
        y_range: tuple[float, float] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.stream_name = stream_name
        self.window_secs = max(window_secs, 0.5)
        self.refresh_hz = max(refresh_hz, 1)
        self.y_range = y_range
        self._plots: list[pg.PlotItem] = []

        streams = resolve_byprop("name", stream_name, timeout=5)
        if not streams:
            raise RuntimeError(f"No LSL stream named '{stream_name}' found.")

        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.sampling_rate = info.nominal_srate() or 250.0
        self.num_channels = info.channel_count()
        self.channel_labels = self._resolve_channel_labels(info)

        self.buffer_samples = max(1, int(self.window_secs * self.sampling_rate))
        self.buffers = np.zeros((self.num_channels, self.buffer_samples))
        self.time_axis = np.linspace(-self.window_secs, 0, self.buffer_samples)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.graphics_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_widget)

        self.curves: list[pg.PlotDataItem] = []
        colors = ["#ff5555", "#55ff55", "#5555ff", "#ffcc00", "#00cccc", "#cc55ff", "#ffffff"]
        for idx in range(self.num_channels):
            plot = self.graphics_widget.addPlot(row=idx, col=0)
            self._plots.append(plot)
            channel_title = self.channel_labels[idx]
            plot.showAxis("left", True)
            is_last = idx == self.num_channels - 1
            plot.showAxis("bottom", is_last)
            if not is_last:
                plot.getAxis("bottom").setVisible(False)
            else:
                plot.setLabel("bottom", "Time (s)")

            plot.setLabel("left", f"{channel_title}")
            if self.y_range:
                plot.setYRange(*self.y_range)
            else:
                plot.enableAutoRange(axis="y")
            curve = plot.plot(pen=pg.mkPen(colors[idx % len(colors)], width=1.5))
            self.curves.append(curve)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_data)
        self.timer.start(int(1000 / self.refresh_hz))

    def _update_data(self) -> None:
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        if not chunk:
            return

        data = np.asarray(chunk).T  # (channels, samples)
        num_samples = data.shape[1]
        if num_samples >= self.buffer_samples:
            self.buffers = data[:, -self.buffer_samples :]
        else:
            self.buffers = np.roll(self.buffers, -num_samples, axis=1)
            self.buffers[:, -num_samples:] = data

        for idx, curve in enumerate(self.curves):
            curve.setData(self.time_axis, self.buffers[idx])

        if not self.y_range:
            self._adjust_scales()

    def _adjust_scales(self) -> None:
        """Compute dynamic min/max across channels and tweak plot ranges with hysteresis."""
        data = self.buffers
        if not data.size:
            return

        for channel_idx, plot in enumerate(self._plots):
            channel = data[channel_idx]
            finite_vals = channel[np.isfinite(channel)]
            if finite_vals.size == 0:
                continue

            channel_min = float(finite_vals.min())
            channel_max = float(finite_vals.max())

            max_abs = max(abs(channel_min), abs(channel_max), 1.0)
            target_max = max_abs * 1.2

            current_range = plot.viewRange()[1]
            current_max = max(abs(current_range[0]), abs(current_range[1]))

            if current_max == 0 or target_max > current_max * 1.4 or target_max < current_max * 0.6:
                self._set_axis_range(plot, -target_max, target_max)

    def _set_axis_range(self, plot: pg.PlotItem,lower: float, upper: float) -> None:
        """Set axis range with nice ticks."""
        plot.setYRange(lower, upper, padding=0.05)

        axis = plot.getAxis("left")
        max_val = max(abs(lower), abs(upper))
        tick_val = max(int(round(max_val)), 1)
        ticks = [
            (-tick_val, f"{-tick_val}"),
            (0.0, "0"),
            (tick_val, f"{tick_val}"),
        ]
        axis.setTicks([ticks])

    def shutdown(self) -> None:
        """Stop timers and release any inlet resources."""
        if self.timer.isActive():
            self.timer.stop()
        try:
            self.inlet.close_stream()
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.shutdown()
        super().closeEvent(event)

    def _resolve_channel_labels(self, info) -> list[str]:
        """Extract readable channel labels from an LSL stream info."""
        labels: list[str] = []
        channels = info.desc().child("channels")
        channel = channels.child("channel")
        while not channel.empty():
            label = channel.child_value("label")
            labels.append(label if label else f"Ch {len(labels) + 1}")
            channel = channel.next_sibling()

        if not labels:
            labels = [f"Ch {idx + 1}" for idx in range(info.channel_count())]

        if len(labels) < info.channel_count():
            labels.extend(
                f"Ch {idx + 1}" for idx in range(len(labels), info.channel_count())
            )

        return labels
