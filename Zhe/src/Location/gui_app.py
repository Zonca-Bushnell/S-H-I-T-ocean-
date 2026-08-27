from __future__ import annotations

import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
from netCDF4 import Dataset, num2date
from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
LOCATION_DIR = SRC_DIR / "Location"
LOCATION_PACKAGE = "src.Location"
BASE_DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config_3d_cmems.yaml"
ACTIVE_CONFIG_MARKER = PROJECT_ROOT / "config" / "active_config.txt"
PAPER_DIR = Path("C:/Users/chenz/Desktop/PAPER")


def load_active_config_path() -> Path:
    try:
        path = Path(ACTIVE_CONFIG_MARKER.read_text(encoding="utf-8").strip())
        if path.exists():
            return path
    except OSError:
        pass
    return BASE_DEFAULT_CONFIG


DEFAULT_CONFIG = load_active_config_path()
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "kuroshio_cmems_3d"
LOG_ROOT = PROJECT_ROOT / "logs" / "gui_runs"


def bi(zh: str, en: str) -> str:
    return f"{zh} / {en}"


def quote_command(parts: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(part) for part in parts])
    return " ".join(shlex.quote(str(part)) for part in parts)


def module_args(module: str) -> list[str]:
    return ["-m", f"{LOCATION_PACKAGE}.{module}"]


def module_file_path(module: str) -> Path:
    return LOCATION_DIR.joinpath(*module.split(".")).with_suffix(".py")


def newest_dir(parent: Path, prefix: str) -> Path:
    candidates = [p for p in parent.glob(f"{prefix}*") if p.is_dir()]
    if not candidates:
        return parent
    return max(candidates, key=lambda p: p.stat().st_mtime)


def config_output_root(config_path: str | Path) -> Path:
    try:
        with Path(config_path).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        value = config.get("paths", {}).get("output_dir")
        if value:
            return Path(value)
    except OSError:
        pass
    return OUTPUT_ROOT


def set_active_config_marker(config_path: Path) -> None:
    ACTIVE_CONFIG_MARKER.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_MARKER.write_text(str(config_path.resolve()), encoding="utf-8")


def open_in_explorer(path: Path) -> None:
    path = path.resolve()
    if path.is_file():
        os.startfile(str(path.parent))
    else:
        os.startfile(str(path))


class PathEdit(QWidget):
    def __init__(self, text: str = "", mode: str = "file") -> None:
        super().__init__()
        self.mode = mode
        self.edit = QLineEdit(text)
        self.button = QPushButton(bi("娴忚", "Browse"))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)
        self.button.clicked.connect(self._browse)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value)

    def _browse(self) -> None:
        start = self.text() or str(PROJECT_ROOT)
        if self.mode == "dir":
            value = QFileDialog.getExistingDirectory(self, "Select directory", start)
        else:
            value, _ = QFileDialog.getOpenFileName(self, bi("閫夋嫨鏂囦欢", "Select file"), start)
        if value:
            self.edit.setText(value)


@dataclass
class TaskSpec:
    name: str
    args: list[str]
    expected_output: Path | None = None
    working_dir: Path = PROJECT_ROOT


class TaskRunner(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.process: QProcess | None = None
        self.started_at: datetime | None = None
        self.current_task: TaskSpec | None = None
        self.current_log: Path | None = None
        self.current_phase = "idle"
        self.current_error = ""
        self.current_pid = ""

        self.command_box = QTextEdit()
        self.command_box.setReadOnly(True)
        self.command_box.setMaximumHeight(80)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.status = QLabel(bi("绌洪棽", "Idle"))
        self.run_button = QPushButton("Run Prepared Command")
        self.stop_button = QPushButton(bi("鍋滄", "Stop"))
        self.stop_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.status, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Prepared command"))
        layout.addWidget(self.command_box)
        layout.addLayout(buttons)
        layout.addWidget(QLabel(bi("浠诲姟鏃ュ織", "Task log")))
        layout.addWidget(self.log_box, 1)

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._tick)
        self.run_button.clicked.connect(self._run_prepared)
        self.stop_button.clicked.connect(self.stop)

        self.prepared: TaskSpec | None = None
        LOG_ROOT.mkdir(parents=True, exist_ok=True)

    def prepare(self, task: TaskSpec) -> None:
        self.prepared = task
        command = quote_command([sys.executable, *task.args])
        self.command_box.setPlainText(command)
        self.log_box.clear()
        self.log_box.insertPlainText(f"[GUI] Prepared task: {task.name}\n")
        self.log_box.insertPlainText(f"[GUI] Working directory: {task.working_dir}\n")
        self.log_box.insertPlainText(f"$ {command}\n")
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)
        self.status.setText(f"Prepared: {task.name}")

    def _run_prepared(self) -> None:
        if self.prepared is None:
            self.current_log = self._new_log_path("no_prepared_command")
            self.log_box.clear()
            self._append_log("[GUI] No prepared command. Click a page-level Prepare Command first.\n")
            self._write_history(exit_code=-1, elapsed="", phase="start_failed", error_text="No prepared command")
            QMessageBox.information(self, "No command", "Please prepare a command first.")
            return
        self.run(self.prepared)

    def run(self, task: TaskSpec) -> None:
        if self.process is not None:
            QMessageBox.warning(self, "Task running", "A task is already running.")
            return
        self.current_log = self._new_log_path(task.name)
        self.current_task = task
        self.started_at = datetime.now()
        self.current_phase = "starting"
        self.current_error = ""
        self.current_pid = ""
        self.log_box.clear()
        self._append_log(f"$ {quote_command([sys.executable, *task.args])}\n")
        self._append_log(f"[GUI] Python: {sys.executable}\n")
        self._append_log(f"[GUI] Working directory: {task.working_dir}\n")
        self._append_log(f"[GUI] Expected output: {task.expected_output or ''}\n")
        validation_error = self._validate_task(task)
        if validation_error:
            self.current_error = validation_error
            self.current_phase = "start_failed"
            self._append_log(f"[GUI] Start failed: {validation_error}\n")
            self._write_history(exit_code=-1, elapsed="0:00:00", phase="start_failed", error_text=validation_error)
            self.status.setText("Start failed")
            self.current_task = None
            QMessageBox.warning(self, "Start failed", validation_error)
            return
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(task.working_dir))
        self.process.setProgram(sys.executable)
        self.process.setArguments(task.args)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._finished)
        self.process.started.connect(self._started)
        self.process.errorOccurred.connect(self._process_error)
        self.process.stateChanged.connect(self._state_changed)
        self.process.start()
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.elapsed_timer.start(1000)
        self.status.setText(f"Starting: {task.name}")
        QTimer.singleShot(1500, self._check_started)

    def stop(self) -> None:
        if self.process is None:
            return
        self._append_log("\n[GUI] Terminating current task...\n")
        self.process.terminate()
        QTimer.singleShot(4000, self._kill_if_running)

    def _new_log_path(self, task_name: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in task_name)
        return LOG_ROOT / f"{stamp}_{safe}.log"

    def _validate_task(self, task: TaskSpec) -> str:
        python_path = Path(sys.executable)
        if not python_path.exists():
            return f"Python executable does not exist: {python_path}"
        if not task.working_dir.exists():
            return f"Working directory does not exist: {task.working_dir}"
        arg_map: dict[str, str] = {}
        for i, value in enumerate(task.args[:-1]):
            if value.startswith("--"):
                arg_map[value] = task.args[i + 1]
        config_arg = arg_map.get("--config")
        if config_arg and not Path(config_arg).exists():
            return f"Config does not exist: {config_arg}"
        if not config_arg and not DEFAULT_CONFIG.exists():
            return f"Default config does not exist: {DEFAULT_CONFIG}"
        if task.args[:1] == ["-m"] and len(task.args) > 1:
            module = task.args[1]
            if module.startswith(f"{LOCATION_PACKAGE}."):
                module_path = module_file_path(module.removeprefix(f"{LOCATION_PACKAGE}."))
                if not module_path.exists():
                    return f"Module entrypoint is missing: {module_path}"
        elif task.args:
            first = task.args[0]
            if first.endswith(".py") and not Path(first).exists():
                return f"Script does not exist: {first}"
        return ""

    def _started(self) -> None:
        if self.process is None or self.current_task is None:
            return
        self.current_phase = "running"
        pid = int(self.process.processId())
        self.current_pid = str(pid)
        self._append_log(f"[GUI] Process started pid={pid}\n")
        self.status.setText(f"Running: {self.current_task.name}, pid={pid}")
        self._write_history(exit_code="", elapsed="", phase="running", error_text="")

    def _state_changed(self, state) -> None:
        names = {
            QProcess.NotRunning: "NotRunning",
            QProcess.Starting: "Starting",
            QProcess.Running: "Running",
        }
        self._append_log(f"[GUI] Process state: {names.get(state, str(state))}\n")

    def _check_started(self) -> None:
        if self.process is None:
            return
        if self.process.state() == QProcess.NotRunning and self.current_phase == "starting":
            error_text = self.process.errorString() or "Process did not start."
            self.current_error = error_text
            self.current_phase = "start_failed"
            self._append_log(f"[GUI] Start failed: {error_text}\n")
            self._write_history(exit_code=-1, elapsed="0:00:00", phase="start_failed", error_text=error_text)
            self._reset_buttons()

    def _process_error(self, error) -> None:
        error_text = self.process.errorString() if self.process is not None else str(error)
        self.current_error = error_text
        if self.current_phase == "starting":
            self.current_phase = "start_failed"
            self._write_history(exit_code=-1, elapsed="0:00:00", phase="start_failed", error_text=error_text)
            self._reset_buttons()
        self._append_log(f"[GUI] Process error: {error_text}\n")

    def _kill_if_running(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _reset_buttons(self) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.elapsed_timer.stop()

    def _append_log(self, text: str) -> None:
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)
        self.log_box.insertPlainText(text)
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)
        if self.current_log is not None:
            with self.current_log.open("a", encoding="utf-8", errors="replace") as f:
                f.write(text)

    def _read_stdout(self) -> None:
        if self.process is not None:
            self._append_log(bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace"))

    def _read_stderr(self) -> None:
        if self.process is not None:
            self._append_log(bytes(self.process.readAllStandardError()).decode("utf-8", "replace"))

    def _finished(self, exit_code: int, exit_status) -> None:
        elapsed = ""
        if self.started_at is not None:
            elapsed = str(datetime.now() - self.started_at).split(".")[0]
        crashed = exit_status == QProcess.CrashExit
        phase = "crashed" if crashed else "finished"
        self.current_phase = phase
        self._append_log(f"\n[GUI] Finished with exit code {exit_code}, phase={phase}, elapsed={elapsed}\n")
        if self.current_task is not None:
            self._write_history(exit_code=exit_code, elapsed=elapsed, phase=phase, error_text=self.current_error)
        self.process = None
        self._reset_buttons()
        self.status.setText(f"Finished: exit={exit_code}, elapsed={elapsed}")

    def _tick(self) -> None:
        if self.started_at is None or self.current_task is None:
            return
        elapsed = str(datetime.now() - self.started_at).split(".")[0]
        self.status.setText(f"{'Running'}: {self.current_task.name}, {bi('鑰楁椂', 'elapsed')}={elapsed}")

    def _write_history(self, exit_code, elapsed: str, phase: str = "finished", error_text: str = "") -> None:
        history = LOG_ROOT / "run_history.csv"
        header = ["time", "task", "phase", "command", "working_dir", "expected_output", "exit_code", "elapsed", "pid", "error_text", "log"]
        new_file = not history.exists()
        if history.exists():
            try:
                with history.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    first = next(csv.reader(f), [])
                if first != header:
                    backup = LOG_ROOT / f"run_history_previous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    history.replace(backup)
                    new_file = True
            except Exception:
                new_file = True
        with history.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(header)
            pid = ""
            if self.process is not None:
                try:
                    pid = self.current_pid or str(int(self.process.processId()))
                except Exception:
                    pid = ""
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    self.current_task.name if self.current_task else "",
                    phase,
                    quote_command([sys.executable, *(self.current_task.args if self.current_task else [])]),
                    str(self.current_task.working_dir if self.current_task else ""),
                    str(self.current_task.expected_output or "") if self.current_task else "",
                    exit_code,
                    elapsed,
                    pid,
                    error_text,
                    str(self.current_log or ""),
                ]
            )


def add_line(form: QFormLayout, label: str, default: str = "") -> QLineEdit:
    edit = QLineEdit(default)
    form.addRow(label, edit)
    return edit


def add_spin(form: QFormLayout, label: str, default: int, min_value: int = 0, max_value: int = 999999) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(min_value, max_value)
    spin.setValue(default)
    form.addRow(label, spin)
    return spin


def add_check(form: QFormLayout, label: str, checked: bool = False) -> QCheckBox:
    box = QCheckBox()
    box.setChecked(checked)
    form.addRow(label, box)
    return box


class CommandPage(QWidget):
    def __init__(self, runner: TaskRunner, title: str) -> None:
        super().__init__()
        self.runner = runner
        self.title = title
        self.form = QFormLayout()
        self.prepare_button = QPushButton(bi("鍑嗗鍛戒护", "Prepare Command"))
        self.run_button = QPushButton("Prepare && Run")
        self.prepare_button.clicked.connect(self.prepare)
        self.run_button.clicked.connect(self.prepare_and_run)

        buttons = QHBoxLayout()
        buttons.addWidget(self.prepare_button)
        buttons.addWidget(self.run_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        box = QGroupBox(title)
        box.setLayout(self.form)
        layout.addWidget(box)
        layout.addLayout(buttons)
        layout.addStretch(1)

    def build_task(self) -> TaskSpec:
        raise NotImplementedError

    def prepare(self) -> TaskSpec | None:
        try:
            task = self.build_task()
        except Exception as exc:
            self.runner.current_log = self.runner._new_log_path(f"{self.title}_prepare_failed")
            self.runner.log_box.clear()
            self.runner._append_log(f"[GUI] Prepare failed on page {self.title}: {exc}\n")
            self.runner._write_history(exit_code=-1, elapsed="", phase="start_failed", error_text=f"Prepare failed: {exc}")
            QMessageBox.warning(self, "Prepare failed", str(exc))
            return None
        self.runner.prepare(task)
        return task

    def prepare_and_run(self) -> None:
        task = self.prepare()
        if task is None:
            return
        self.runner.run(task)


class PipelinePage(CommandPage):
    def __init__(self, runner: TaskRunner) -> None:
        super().__init__(runner, bi("瀹屾暣涓夌淮娴佺▼", "Full 3D Pipeline"))
        self.config = PathEdit(str(DEFAULT_CONFIG))
        self.form.addRow(bi("閰嶇疆", "Config"), self.config)
        self.start = add_line(self.form, "Start date", "1993-01-01")
        self.end = add_line(self.form, bi("缁撴潫鏃ユ湡", "End date"), "2022-12-31")
        self.depth_index = add_line(self.form, "Depth index optional", "")
        self.skip_convert = add_check(self.form, bi("璺宠繃杞崲", "Skip convert"), False)
        self.skip_identify = add_check(self.form, bi("璺宠繃璇嗗埆", "Skip identify"), False)
        self.skip_link = add_check(self.form, bi("璺宠繃鍨傚悜杩炴帴", "Skip link"), False)
        self.skip_track = add_check(self.form, bi("璺宠繃杩借釜", "Skip track"), False)
        self.skip_export = add_check(self.form, bi("璺宠繃瀵煎嚭", "Skip export"), False)

    def build_task(self) -> TaskSpec:
        args = [
            *module_args("run_3d_pipeline"),
            "--config",
            self.config.text(),
            "--start",
            self.start.text(),
            "--end",
            self.end.text(),
            "--workers",
            str(self.workers.value()),
        ]
        if self.force.isChecked():
            args.append("--force")
        if self.depth_index.text():
            args += ["--depth-index", self.depth_index.text()]
        skips = []
        for name, box in (
            ("convert", self.skip_convert),
            ("identify", self.skip_identify),
            ("link", self.skip_link),
            ("track", self.skip_track),
            ("export", self.skip_export),
        ):
            if box.isChecked():
                skips.append(name)
        if skips:
            args += ["--skip", *skips]
        return TaskSpec("pipeline", args, config_output_root(self.config.text()) / "catalog")


class ShapePage(CommandPage):
    def __init__(self, runner: TaskRunner) -> None:
        super().__init__(runner, "Shape Classification")
        self.config = PathEdit(str(DEFAULT_CONFIG))
        self.form.addRow(bi("閰嶇疆", "Config"), self.config)
        self.start = add_line(self.form, "Start date", "1993-01-01")
        self.end = add_line(self.form, bi("缁撴潫鏃ユ湡", "End date"), "2022-12-31")
        self.output_name = add_line(self.form, bi("杈撳嚭鍚嶇О", "Output name"), "shape_classification_1993_2022")
        self.lifetime = add_spin(self.form, "Lifetime min days", 56, 0, 10000)
        self.radius = add_line(self.form, bi("鏈€灏忓崐寰?m", "Radius min m"), "50000")
        self.min_layers = add_spin(self.form, bi("鏈€灏戞湁鏁堝眰", "Min valid layers"), 6, 1, 200)
        self.complete_missing = add_check(self.form, bi("琛ュ叏缂哄け鏍稿績", "Complete missing centers"), True)
        self.centers_only = add_check(self.form, bi("鍙ˉ鏍稿績", "Centers only"), True)

    def build_task(self) -> TaskSpec:
        args = [
            *module_args("classify_3d_eddy_shape"),
            "--config",
            self.config.text(),
            "--workers",
            str(self.workers.value()),
            "--lifetime-min-days",
            str(self.lifetime.value()),
            "--radius-min-m",
            self.radius.text(),
            "--min-valid-layers",
            str(self.min_layers.value()),
            "--start",
            self.start.text(),
            "--end",
            self.end.text(),
            "--output-name",
            self.output_name.text(),
        ]
        args.append("--complete-missing" if self.complete_missing.isChecked() else "--no-complete-missing")
        args += ["--completion-output-mode", "centers-only" if self.centers_only.isChecked() else "centers-and-contours"]
        if self.force.isChecked():
            args.append("--force")
        return TaskSpec("shape_classification", args, config_output_root(self.config.text()) / self.output_name.text())


class CompositePage(CommandPage):
    def __init__(self, runner: TaskRunner) -> None:
        super().__init__(runner, bi("鐢熷懡鍛ㄦ湡鍚堟垚", "Lifecycle Composite"))
        default_shape_dir = newest_dir(config_output_root(DEFAULT_CONFIG), "shape_classification_")
        self.config = PathEdit(str(DEFAULT_CONFIG))
        self.shape_dir = PathEdit(str(default_shape_dir), "dir")
        self.form.addRow(bi("閰嶇疆", "Config"), self.config)
        self.form.addRow(bi("鍒嗙被鐩綍", "Shape dir"), self.shape_dir)
        self.start = add_line(self.form, "Start date", "1993-01-01")
        self.end = add_line(self.form, bi("缁撴潫鏃ユ湡", "End date"), "2022-12-31")
        self.shape = add_line(self.form, "Shape", "coherent")
        self.phase_bins = add_spin(self.form, "Phase bins", 5, 1, 20)
        self.rmax = add_line(self.form, bi("鍗婂緞鑼冨洿 Rmax", "R max"), "2.0")
        self.ngrid = add_spin(self.form, bi("缃戞牸澶у皬", "Grid size"), 61, 11, 401)
        self.max_tracks = add_line(self.form, "Max tracks per group optional", "")
        self.include_sigma0 = add_check(self.form, bi("鍖呭惈浣嶅娍瀵嗗害 sigma0", "Include sigma0"), True)
        self.force_clim = add_check(self.form, "Force climatology", False)

    def build_task(self) -> TaskSpec:
        args = [
            *module_args("composite_3d_lifecycle"),
            "--config",
            self.config.text(),
            "--shape-dir",
            self.shape_dir.text(),
            "--start",
            self.start.text(),
            "--end",
            self.end.text(),
            "--shape",
            self.shape.text(),
            "--phase-bins",
            str(self.phase_bins.value()),
            "--group-by",
            "shape",
            "polarity",
            "--rmax",
            self.rmax.text(),
            "--ngrid",
            str(self.ngrid.value()),
            "--workers",
            str(self.workers.value()),
            "--anomaly-source",
            "doy-climatology",
            "--climatology-smooth-days",
            "31",
        ]
        if self.max_tracks.text():
            args += ["--max-tracks-per-group", self.max_tracks.text()]
        if self.include_sigma0.isChecked():
            args.append("--include-sigma0")
        if self.force_clim.isChecked():
            args.append("--force-climatology")
        out = config_output_root(self.config.text()) / f"lifecycle_composites_{self.start.text()[:4]}_{self.end.text()[:4]}_{self.shape.text()}"
        return TaskSpec("lifecycle_composite", args, out)


class PlotCompositePage(CommandPage):
    def __init__(self, runner: TaskRunner) -> None:
        super().__init__(runner, bi("缁樺埗鐢熷懡鍛ㄦ湡鍚堟垚", "Plot Lifecycle Composite"))
        default_comp = newest_dir(config_output_root(DEFAULT_CONFIG), "lifecycle_composites_")
        self.input_dir = PathEdit(str(default_comp), "dir")
        self.topview_dir = PathEdit("", "dir")
        self.form.addRow(bi("鍚堟垚鐩綍", "Composite dir"), self.input_dir)
        self.variables = add_line(self.form, bi("鍙橀噺", "Variables"), "thetao_anom,so_anom,sigma0_anom,adt_anom,mlotst_anom,speed_anom")
        self.depths = add_line(self.form, bi("娣卞害 m", "Depths m"), "0,100,300,700,1000,1500")
        self.polarities = add_line(self.form, "Polarities optional", "")
        self.plot_summary = add_check(self.form, bi("缁樺埗姒傝", "Plot summary"), False)
        self.plot_horizontal = add_check(self.form, "Plot horizontal", True)
        self.plot_sections = add_check(self.form, "Plot sections", True)
        self.plot_coverage = add_check(self.form, "Plot coverage", True)
        self.plot_topview = add_check(self.form, "Plot topview", False)
        self.topview_all_depths = add_check(self.form, "Topview all depths", False)
        self.no_arrows = add_check(self.form, bi("涓嶇敾绠ご", "No arrows"), False)
        self.arrow_step = add_spin(self.form, bi("绠ご闂撮殧", "Arrow step"), 5, 1, 100)
        self.form.addRow("Topview output dir", self.topview_dir)

    def build_task(self) -> TaskSpec:
        plots = []
        if self.plot_summary.isChecked():
            plots.append("summary")
        if self.plot_horizontal.isChecked():
            plots.append("horizontal")
        if self.plot_sections.isChecked():
            plots.append("sections")
        if self.plot_coverage.isChecked():
            plots.append("coverage")
        if self.plot_topview.isChecked():
            plots.append("topview")
        if not plots:
            plots = ["horizontal"]
        args = [
            *module_args("plot_lifecycle_composite"),
            "--input-dir",
            self.input_dir.text(),
            "--variables",
            self.variables.text(),
            "--depths",
            self.depths.text(),
            "--plot",
            ",".join(plots),
            "--arrow-step",
            str(self.arrow_step.value()),
        ]
        if self.polarities.text():
            args += ["--polarities", self.polarities.text()]
        if self.no_arrows.isChecked():
            args.append("--no-arrows")
        if self.topview_all_depths.isChecked():
            args.append("--topview-all-depths")
        if self.topview_dir.text():
            args += ["--topview-output-dir", self.topview_dir.text()]
        expected = Path(self.topview_dir.text()) if self.topview_dir.text() else Path(self.input_dir.text()) / "figures"
        return TaskSpec("plot_composite", args, expected)


class SingleIdPage(CommandPage):
    def __init__(self, runner: TaskRunner) -> None:
        super().__init__(runner, "Single Track3D Daily 3D")
        self.config = PathEdit(str(DEFAULT_CONFIG))
        self.form.addRow(bi("閰嶇疆", "Config"), self.config)
        self.track_id = add_line(self.form, "track3d_id", "2")
        self.output_dir = PathEdit("", "dir")
        self.form.addRow("Output dir optional", self.output_dir)
        self.fps = add_spin(self.form, "甯х巼 FPS", 2, 1, 60)
        self.elev = add_line(self.form, "Elev", "32")
        self.azim = add_line(self.form, "Azim", "-62")
        self.vertical_exaggeration = add_line(self.form, "Vertical exaggeration", "20")
        self.padding = add_line(self.form, "Padding km", "60")
        self.show_all_depths = add_check(self.form, bi("鏄剧ず鍏ㄩ儴娣卞害", "Show all depths"), True)
        self.show_speed_fill = add_check(self.form, bi("鏄剧ず閫熷害濉壊", "Show speed fill"), False)
        self.show_arrows = add_check(self.form, bi("鏄剧ず绠ご", "Show arrows"), False)

    def build_task(self) -> TaskSpec:
        args = [
            *module_args("plot_3d_eddy_daily"),
            "--config",
            self.config.text(),
            "--track3d-id",
            self.track_id.text(),
            "--fps",
            str(self.fps.value()),
            "--elev",
            self.elev.text(),
            "--azim",
            self.azim.text(),
            "--vertical-exaggeration",
            self.vertical_exaggeration.text(),
            "--padding-km",
            self.padding.text(),
        ]
        if self.output_dir.text():
            args += ["--output-dir", self.output_dir.text()]
        if self.show_all_depths.isChecked():
            args.append("--show-all-depths")
        if self.show_speed_fill.isChecked():
            args.append("--show-speed-fill")
        if self.show_arrows.isChecked():
            args.append("--show-arrows")
        if self.force.isChecked():
            args.append("--force")
        expected = Path(self.output_dir.text()) if self.output_dir.text() else config_output_root(self.config.text()) / "figures" / f"track3d_{self.track_id.text()}"
        return TaskSpec("single_id_3d", args, expected)


class ClimatologyPage(CommandPage):
    def __init__(self, runner: TaskRunner) -> None:
        super().__init__(runner, "CMEMS Day-of-Year Climatology")
        self.config = PathEdit(str(DEFAULT_CONFIG))
        self.form.addRow(bi("閰嶇疆", "Config"), self.config)
        self.start = add_line(self.form, "Start date", "1993-01-01")
        self.end = add_line(self.form, bi("缁撴潫鏃ユ湡", "End date"), "2022-12-31")
        self.smooth = add_spin(self.form, bi("骞虫粦澶╂暟", "Smooth days"), 31, 1, 365)

    def build_task(self) -> TaskSpec:
        args = [
            *module_args("build_cmems_climatology"),
            "--config",
            self.config.text(),
            "--start",
            self.start.text(),
            "--end",
            self.end.text(),
            "--smooth-days",
            str(self.smooth.value()),
        ]
        if self.force.isChecked():
            args.append("--force")
        return TaskSpec("climatology", args, config_output_root(self.config.text()) / "climatology")


class DataSourcePage(QWidget):
    def __init__(self, runner: TaskRunner, on_apply) -> None:
        super().__init__()
        self.runner = runner
        self.on_apply = on_apply
        self.files: list[Path] = []
        acc_root = Path("E:/DATA/Copernicus_Data/ACC")

        self.source_path = PathEdit(str(acc_root if acc_root.exists() else PROJECT_ROOT), "dir")
        self.project_name = QLineEdit("acc_cmems_3d")
        self.output_root = PathEdit(str(PROJECT_ROOT / "outputs" / "acc_cmems_3d"), "dir")
        self.config_output = QLineEdit(str(PROJECT_ROOT / "config" / "config_3d_acc.yaml"))
        self.bbox = QLineEdit("-179,180,-65,-45")
        self.max_depth = QLineEdit("2000")
        self.start_date = QLineEdit("2018-01-01")
        self.end_date = QLineEdit("2019-12-31")
        self.lon_name = QLineEdit("longitude")
        self.lat_name = QLineEdit("latitude")
        self.depth_name = QLineEdit("depth")
        self.time_name = QLineEdit("time")
        self.u_name = QLineEdit("uo_glor")
        self.v_name = QLineEdit("vo_glor")
        self.height_name = QLineEdit("zos_glor")

        self.scan_button = QPushButton("Scan metadata")
        self.generate_button = QPushButton("Generate config and activate")
        self.metadata = QTextEdit()
        self.metadata.setReadOnly(True)

        form = QFormLayout()
        form.addRow("NetCDF file or directory", self.source_path)
        self.include_part = QCheckBox()
        self.include_part.setChecked(False)
        form.addRow("Include *.part.nc", self.include_part)
        form.addRow("Project name", self.project_name)
        form.addRow("Output root", self.output_root)
        form.addRow("Config output", self.config_output)
        form.addRow("bbox lon0,lon1,lat0,lat1", self.bbox)
        form.addRow("Max depth m", self.max_depth)
        form.addRow("Start date", self.start_date)
        form.addRow("End date", self.end_date)
        form.addRow("lon/lat/depth/time", self._row_widget([self.lon_name, self.lat_name, self.depth_name, self.time_name]))
        form.addRow("u/v/height", self._row_widget([self.u_name, self.v_name, self.height_name]))

        buttons = QHBoxLayout()
        buttons.addWidget(self.scan_button)
        buttons.addWidget(self.generate_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        box = QGroupBox("Data Source / Metadata")
        box.setLayout(form)
        layout.addWidget(box)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Metadata preview"))
        layout.addWidget(self.metadata, 1)

        self.scan_button.clicked.connect(self.scan_metadata)
        self.generate_button.clicked.connect(self.generate_config)

    @staticmethod
    def _row_widget(widgets: list[QLineEdit]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        for item in widgets:
            layout.addWidget(item)
        return widget

    def _discover_files(self) -> list[Path]:
        path = Path(self.source_path.text())
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(path.glob("*.nc"))
        else:
            raise FileNotFoundError(f"Source path does not exist: {path}")
        if not self.include_part.isChecked():
            files = [item for item in files if ".part." not in item.name.lower() and not item.name.lower().endswith(".part.nc")]
        if not files:
            raise FileNotFoundError("No NetCDF files found after filtering.")
        return files

    def scan_metadata(self) -> None:
        try:
            files = self._discover_files()
            summaries = [self._scan_one(path) for path in files]
        except Exception as exc:
            self.metadata.setPlainText(f"[scan failed] {exc}")
            QMessageBox.warning(self, "Scan failed", str(exc))
            return
        self.files = files
        lon0 = min(item["lon_min"] for item in summaries)
        lon1 = max(item["lon_max"] for item in summaries)
        lat0 = min(item["lat_min"] for item in summaries)
        lat1 = max(item["lat_max"] for item in summaries)
        max_depth = max(item["depth_max"] for item in summaries)
        start = min(item["start"] for item in summaries)
        end = max(item["end"] for item in summaries)
        self.bbox.setText(f"{lon0:g},{lon1:g},{lat0:g},{lat1:g}")
        is_acc = "acc" in str(Path(self.source_path.text())).lower()
        self.max_depth.setText("2000" if is_acc else f"{max_depth:.10g}")
        self.start_date.setText(start)
        self.end_date.setText(end)
        if is_acc:
            self.project_name.setText("acc_cmems_3d")
            self.output_root.setText(str(PROJECT_ROOT / "outputs" / "acc_cmems_3d"))
            self.config_output.setText(str(PROJECT_ROOT / "config" / "config_3d_acc.yaml"))
        lines = [
            f"Files selected: {len(files)}",
            f"Combined bbox: {self.bbox.text()}",
            f"Date range: {start} -> {end}",
            f"Depth max: {max_depth:g} m",
            "",
        ]
        for item in summaries:
            partial = " PARTIAL" if item["partial"] else ""
            lines.append(f"{item['path']}{partial}")
            lines.append(f"  dims={item['dims']}")
            lines.append(f"  time={item['start']} -> {item['end']} n={item['time_count']}")
            lines.append(f"  lon={item['lon_min']:g}..{item['lon_max']:g}, lat={item['lat_min']:g}..{item['lat_max']:g}, depth={item['depth_min']:g}..{item['depth_max']:g}")
        self.metadata.setPlainText("\n".join(lines))

    def _scan_one(self, path: Path) -> dict:
        with Dataset(path) as ds:
            lon = np.asarray(ds.variables[self.lon_name.text()][:], dtype="f8")
            lat = np.asarray(ds.variables[self.lat_name.text()][:], dtype="f8")
            depth = np.asarray(ds.variables[self.depth_name.text()][:], dtype="f8")
            time_var = ds.variables[self.time_name.text()]
            raw_time = time_var[:]
            dates = num2date(
                raw_time[[0, -1]],
                time_var.units,
                getattr(time_var, "calendar", "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            required = [self.u_name.text(), self.v_name.text(), self.height_name.text(), "thetao_glor", "so_glor", "mlotst_glor"]
            missing = [name for name in required if name not in ds.variables]
            if missing:
                raise KeyError(f"{path} missing variables: {missing}")
            return {
                "path": str(path),
                "dims": {name: len(dim) for name, dim in ds.dimensions.items()},
                "start": dates[0].date().isoformat(),
                "end": dates[-1].date().isoformat(),
                "time_count": int(len(raw_time)),
                "partial": ".part." in path.name.lower() or path.name.lower().endswith(".part.nc") or int(len(raw_time)) < 300,
                "lon_min": float(np.nanmin(lon)),
                "lon_max": float(np.nanmax(lon)),
                "lat_min": float(np.nanmin(lat)),
                "lat_max": float(np.nanmax(lat)),
                "depth_min": float(np.nanmin(depth)),
                "depth_max": float(np.nanmax(depth)),
            }

    def generate_config(self) -> None:
        try:
            if not self.files:
                self.scan_metadata()
            files = self.files or self._discover_files()
            config_path = Path(self.config_output.text())
            output_root = Path(self.output_root.text())
            bbox = [float(item.strip()) for item in self.bbox.text().split(",")]
            if len(bbox) != 4:
                raise ValueError("bbox must contain lon0,lon1,lat0,lat1")
            base = yaml.safe_load(BASE_DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
            file_strings = [str(path).replace("\\", "/") for path in files]
            base["project"]["name"] = self.project_name.text().strip() or "cmems_eddy3d"
            base["data_source"] = {
                "kind": "cmems_netcdf_timeseries",
                "input_nc_file": file_strings[0],
                "input_nc_files": file_strings,
            }
            base["paths"] = {
                "input_mat_dir": "",
                "output_dir": str(output_root).replace("\\", "/"),
                "input_daily_dir": str(output_root / "input_daily").replace("\\", "/"),
                "layer_dir": str(output_root / "layers").replace("\\", "/"),
                "catalog_dir": str(output_root / "catalog").replace("\\", "/"),
                "logs_dir": str(PROJECT_ROOT / "logs").replace("\\", "/"),
            }
            base["region"]["bbox"] = bbox
            base["region"]["max_depth_m"] = float(self.max_depth.text())
            base["date_range"] = {"start": self.start_date.text(), "end": self.end_date.text()}
            base["variables"].update(
                {
                    "source_lon": self.lon_name.text(),
                    "source_lat": self.lat_name.text(),
                    "source_depth": self.depth_name.text(),
                    "source_time": self.time_name.text(),
                    "source_height": self.height_name.text(),
                    "source_u": self.u_name.text(),
                    "source_v": self.v_name.text(),
                    "output_height": "adt",
                }
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8")
            set_active_config_marker(config_path)
            self.on_apply(config_path, output_root)
            self.metadata.append(f"\n[generated] {config_path}")
            self.metadata.append(f"[active output] {output_root}")
            QMessageBox.information(self, "Config generated", f"Generated and activated:\n{config_path}")
        except Exception as exc:
            self.metadata.append(f"\n[generate failed] {exc}")
            QMessageBox.warning(self, "Generate failed", str(exc))


class ValidationPage(CommandPage):
    def __init__(self, runner: TaskRunner) -> None:
        super().__init__(runner, "Theory Validation")
        self.config = PathEdit(str(DEFAULT_CONFIG))
        self.command = QComboBox()
        self.command.addItems(["model-b", "isopycnal-a", "check-model-definitions"])
        self.shape = add_line(self.form, "Single shape", "coherent")
        self.output_dir = PathEdit("", "dir")
        self.form.insertRow(0, "Command", self.command)
        self.form.insertRow(1, "Config", self.config)
        self.form.addRow("Output dir optional", self.output_dir)
        self.depths = add_line(self.form, "Velocity depths", "0,100,300,700,1000,1500")
        self.quick = add_check(self.form, "Quick", False)
        self.plot_sections = add_check(self.form, "Plot sections", True)
        self.plot_topview = add_check(self.form, "Plot topview", False)
        self.plot_tilt_growth = add_check(self.form, "Plot tilt growth", False)

        self.open_papers = QPushButton("Open PAPER folder")
        self.open_paper_index = QPushButton("Open paper_index.csv")
        paper_buttons = QHBoxLayout()
        paper_buttons.addWidget(self.open_papers)
        paper_buttons.addWidget(self.open_paper_index)
        paper_buttons.addStretch(1)
        self.layout().addLayout(paper_buttons)
        self.open_papers.clicked.connect(lambda: open_in_explorer(PAPER_DIR))
        self.open_paper_index.clicked.connect(lambda: open_in_explorer(PAPER_DIR / "paper_index.csv"))

    def build_task(self) -> TaskSpec:
        command = self.command.currentText()
        if command == "check-model-definitions":
            args = [*module_args("forecast.cli"), command, "--forecast-root", str(LOCATION_DIR / "forecast")]
            expected = LOCATION_DIR / "forecast" / "MODEL_DEFINITION_AUDIT_OK.txt"
        else:
            args = [*module_args("forecast.cli"), command, "--config", self.config.text()]
            args += [
                "--shape", self.shape.text(),
                "--depths", self.depths.text(),
            ]
            if self.plot_sections.isChecked():
                args.append("--plot-sections")
            if self.plot_topview.isChecked():
                args.append("--plot-topview")
            if self.plot_tilt_growth.isChecked():
                args.append("--plot-tilt-growth")
            branch = "baseline_A_model_B" if command == "model-b" else "li_baseline_vs_isopycnal_A"
            expected = config_output_root(self.config.text()) / "forecast" / branch / self.shape.text()
        if self.quick.isChecked() and command in {"isopycnal-a", "model-b"}:
            args.append("--quick")
        if self.output_dir.text():
            args += ["--output-dir", self.output_dir.text()]
            expected = Path(self.output_dir.text())
        return TaskSpec(f"validation_{command}", args, expected)


class DashboardPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.refresh_button = QPushButton(bi("鍒锋柊", "Refresh"))
        self.open_outputs = QPushButton(bi("鎵撳紑杈撳嚭", "Open outputs"))
        self.open_logs = QPushButton(bi("鎵撳紑鏃ュ織", "Open logs"))
        self.refresh_button.clicked.connect(self.refresh)
        self.open_outputs.clicked.connect(lambda: open_in_explorer(config_output_root(load_active_config_path())))
        self.open_logs.clicked.connect(lambda: open_in_explorer(PROJECT_ROOT / "logs"))

        buttons = QHBoxLayout()
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.open_outputs)
        buttons.addWidget(self.open_logs)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.info, 1)
        self.refresh()

    def refresh(self) -> None:
        active_config = load_active_config_path()
        active_output_root = config_output_root(active_config)
        rows = [
            f"Project root: {PROJECT_ROOT}",
            f"Python: {sys.executable}",
            f"Active config: {active_config} [{'OK' if active_config.exists() else 'MISSING'}]",
            f"Active output root: {active_output_root} [{'OK' if active_output_root.exists() else 'MISSING'}]",
            f"Base config: {BASE_DEFAULT_CONFIG} [{'OK' if BASE_DEFAULT_CONFIG.exists() else 'MISSING'}]",
            "",
            "Important outputs:",
        ]
        checks = [
            active_output_root / "input_daily",
            active_output_root / "layers",
            active_output_root / "catalog" / "layer_observations.parquet",
            active_output_root / "catalog" / "vertical_objects.parquet",
            active_output_root / "catalog" / "tracks_3d.parquet",
            active_output_root / "catalog" / "layer_centers_completed.parquet",
            active_output_root / "forecast" / "baseline_A_model_B" / "coherent" / "forecast_state.nc",
            active_output_root / "forecast" / "baseline_A_model_B" / "coherent" / "tilt_growth_skill.parquet",
            active_output_root / "forecast" / "baseline_A_model_B" / "coherent" / "validation_summary.md",
        ]
        for path in checks:
            rows.append(self._format_path(path))
        rows.append("")
        rows.append("Recent GUI runs:")
        rows.extend(self._recent_history())
        rows.append("")
        rows.append("Recent result directories:")
        if active_output_root.exists():
            for path in sorted([p for p in active_output_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:18]:
                rows.append(self._format_path(path))
        self.info.setPlainText("\n".join(rows))

    @staticmethod
    def _recent_history() -> list[str]:
        history = LOG_ROOT / "run_history.csv"
        if not history.exists():
            return ["[none] No GUI run history yet."]
        try:
            with history.open("r", encoding="utf-8", errors="replace", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as exc:
            return [f"[history unreadable] {exc}"]
        if not rows:
            return ["[none] History is empty."]
        out: list[str] = []
        for row in rows[-5:][::-1]:
            phase = row.get("phase", "")
            task = row.get("task", "")
            code = row.get("exit_code", "")
            error = row.get("error_text", "")
            stamp = row.get("time", "")
            log = row.get("log", "")
            suffix = f" | error={error}" if error else ""
            out.append(f"[{phase}] {stamp} | {task} | exit={code} | log={log}{suffix}")
        return out

    @staticmethod
    def _format_path(path: Path) -> str:
        if path.exists():
            stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            if path.is_file():
                size = f"{path.stat().st_size / (1024 * 1024):.1f} MB"
            else:
                size = "dir"
            return f"[OK] {path} | {size} | {stamp}"
        return f"[MISSING] {path}"


class ResultsBrowser(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.root_edit = PathEdit(str(config_output_root(DEFAULT_CONFIG)), "dir")
        self.set_root_button = QPushButton("Set Root")
        self.open_button = QPushButton("Open Selected Folder")
        self.model = QFileSystemModel(self)
        self.model.setRootPath(str(config_output_root(DEFAULT_CONFIG)))
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(str(config_output_root(DEFAULT_CONFIG))))
        self.tree.setColumnWidth(0, 420)
        self.preview = QLabel(bi("PNG棰勮", "PNG preview"))
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumWidth(420)
        self.preview.setStyleSheet("QLabel { background: #202020; color: #cccccc; }")

        top = QHBoxLayout()
        top.addWidget(QLabel("Root"))
        top.addWidget(self.root_edit, 1)
        top.addWidget(self.set_root_button)
        top.addWidget(self.open_button)

        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)

        self.set_root_button.clicked.connect(self.set_root)
        self.open_button.clicked.connect(self.open_selected)
        self.tree.selectionModel().currentChanged.connect(self.preview_selected)

    def set_root(self) -> None:
        root = Path(self.root_edit.text())
        if not root.exists():
            QMessageBox.warning(self, bi("鏍圭洰褰曚笉瀛樺湪", "Missing root"), f"{'Path does not exist'}:\n{root}")
            return
        self.model.setRootPath(str(root))
        self.tree.setRootIndex(self.model.index(str(root)))

    def set_root_path(self, root: Path) -> None:
        self.root_edit.setText(str(root))
        if root.exists():
            self.model.setRootPath(str(root))
            self.tree.setRootIndex(self.model.index(str(root)))

    def selected_path(self) -> Path | None:
        index = self.tree.currentIndex()
        if not index.isValid():
            return None
        return Path(self.model.filePath(index))

    def open_selected(self) -> None:
        path = self.selected_path()
        if path is None:
            return
        open_in_explorer(path)

    def preview_selected(self) -> None:
        path = self.selected_path()
        if path is None or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            self.preview.setText(bi("PNG棰勮", "PNG preview"))
            self.preview.setPixmap(QPixmap())
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.preview.setText(bi("鏃犳硶鍔犺浇鍥剧墖", "Could not load image"))
            return
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.preview_selected()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(bi("??/ACC ???????", "3D Eddy Workbench"))
        self.resize(1450, 920)
        self.runner = TaskRunner()
        self.tabs = QTabWidget()

        self.dashboard_page = DashboardPage()
        self.pipeline_page = PipelinePage(self.runner)
        self.climatology_page = ClimatologyPage(self.runner)
        self.shape_page = ShapePage(self.runner)
        self.composite_page = CompositePage(self.runner)
        self.plot_page = PlotCompositePage(self.runner)
        self.validation_page = ValidationPage(self.runner)
        self.single_id_page = SingleIdPage(self.runner)
        self.results_browser = ResultsBrowser()
        self.data_source_page = DataSourcePage(self.runner, self.apply_active_config)

        self.tabs.addTab(self.dashboard_page, bi("??", "Dashboard"))
        self.tabs.addTab(self.data_source_page, bi("???", "Data Source"))
        self.tabs.addTab(self.pipeline_page, bi("????", "Full Pipeline"))
        self.tabs.addTab(self.climatology_page, bi("???", "Climatology"))
        self.tabs.addTab(self.shape_page, bi("????", "Shape"))
        self.tabs.addTab(self.composite_page, bi("??????", "Composite"))
        self.tabs.addTab(self.plot_page, bi("??", "Plot"))
        self.tabs.addTab(self.validation_page, bi("????", "Validation"))
        self.tabs.addTab(self.single_id_page, bi("?ID??", "Single ID 3D"))
        self.tabs.addTab(self.results_browser, bi("????", "Results"))

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.runner)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)
        active_config = load_active_config_path()
        self.apply_active_config(active_config, config_output_root(active_config), announce=False)

    def apply_active_config(self, config_path: Path, output_root: Path, announce: bool = True) -> None:
        set_active_config_marker(config_path)
        date_range = {}
        try:
            loaded = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            date_range = loaded.get("date_range", {}) or {}
            output_root = config_output_root(config_path)
        except Exception:
            pass
        for page in [
            self.pipeline_page,
            self.climatology_page,
            self.shape_page,
            self.composite_page,
            self.validation_page,
            self.single_id_page,
        ]:
            if hasattr(page, "config"):
                page.config.setText(str(config_path))
            if hasattr(page, "start") and date_range.get("start"):
                page.start.setText(str(date_range["start"]))
            if hasattr(page, "end") and date_range.get("end"):
                page.end.setText(str(date_range["end"]))
        self.composite_page.shape_dir.setText(str(newest_dir(output_root, "shape_classification_")))
        self.plot_page.input_dir.setText(str(newest_dir(output_root, "lifecycle_composites_")))
        self.results_browser.set_root_path(output_root)
        self.dashboard_page.refresh()
        if announce:
            self.runner.log_box.clear()
            self.runner.log_box.insertPlainText(f"[GUI] Active config: {config_path}\n[GUI] Active output root: {output_root}\n")

    def closeEvent(self, event) -> None:
        if self.runner.process is not None:
            reply = QMessageBox.question(
                self,
                bi("?????", "Task running"),
                bi("?????????????????", "A task is still running. Stop it and close?"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.stop()
        event.accept()

def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

