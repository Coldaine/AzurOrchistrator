"""Main PySide6 UI application."""

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from PySide6.QtCore import QObject, QTimer, Signal, QThread
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QCheckBox, QSplitter, QProgressBar, QStatusBar
)
import cv2
import numpy as np

from ..core.bootstrap import bootstrap_from_config, create_default_config
from ..core.capture import Frame
from ..core.configs import AppConfig
from .overlays import OverlayRenderer
from .state import UIState


class TaskWorker(QObject):
    """Worker for running tasks in background thread."""
    
    finished = Signal(bool)  # success
    progress = Signal(str)   # status message
    frame_captured = Signal(object)  # Frame object
    
    def __init__(self, planner, task) -> None:
        super().__init__()
        self.planner = planner
        self.task = task
        self._running = False
    
    def run_task(self) -> None:
        """Run the task."""
        self._running = True
        self.progress.emit(f"Starting task: {self.task.name}")
        
        try:
            success = self.planner.run_task(self.task)
            self.finished.emit(success)
        except Exception as e:
            logger.error(f"Task failed: {e}")
            self.progress.emit(f"Task failed: {e}")
            self.finished.emit(False)
        finally:
            self._running = False
    
    def stop(self) -> None:
        """Stop the task (placeholder for future implementation)."""
        self._running = False


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self) -> None:
        super().__init__()
        self.components: Optional[Dict[str, Any]] = None
        self.ui_state = UIState()
        self.overlay_renderer = OverlayRenderer()
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[TaskWorker] = None
        
        self.init_ui()
        self.init_timer()
        
        # Try to load configuration and bootstrap
        self.load_configuration()
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Azur Lane Bot")
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        splitter = QSplitter()
        layout.addWidget(splitter)
        
        # Left panel - Live view
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Controls and data
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (60% left, 40% right)
        splitter.setSizes([840, 560])
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_left_panel(self) -> QWidget:
        """Create the left panel with live frame view."""
        panel = QGroupBox("Live View")
        layout = QVBoxLayout(panel)
        
        # Frame display
        self.frame_label = QLabel("No frame captured")
        self.frame_label.setMinimumSize(640, 360)
        self.frame_label.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        self.frame_label.setScaledContents(True)
        layout.addWidget(self.frame_label)
        
        # Frame info
        self.frame_info = QLabel("Frame info: N/A")
        layout.addWidget(self.frame_info)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with controls and data."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Controls
        controls = self.create_controls_group()
        layout.addWidget(controls)
        
        # Status
        status = self.create_status_group()
        layout.addWidget(status)
        
        # Data tables
        data = self.create_data_group()
        layout.addWidget(data)
        
        # LLM JSON view (collapsible)
        llm_group = self.create_llm_group()
        layout.addWidget(llm_group)
        
        return panel
    
    def create_controls_group(self) -> QGroupBox:
        """Create the controls group."""
        group = QGroupBox("Controls")
        layout = QVBoxLayout(group)
        
        # Task selection
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Task:"))
        
        self.task_combo = QComboBox()
        self.task_combo.addItems(["currencies", "pickups", "commissions"])
        task_layout.addWidget(self.task_combo)
        
        layout.addLayout(task_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_task)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_task)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.run_once_button = QPushButton("Run Once")
        self.run_once_button.clicked.connect(self.run_once)
        button_layout.addWidget(self.run_once_button)
        
        layout.addLayout(button_layout)
        
        # Options
        self.overlay_checkbox = QCheckBox("Show Overlays")
        self.overlay_checkbox.setChecked(True)
        layout.addWidget(self.overlay_checkbox)
        
        return group
    
    def create_status_group(self) -> QGroupBox:
        """Create the status group."""
        group = QGroupBox("Status")
        layout = QVBoxLayout(group)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(120)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        return group
    
    def create_data_group(self) -> QGroupBox:
        """Create the data display group."""
        group = QGroupBox("Data")
        layout = QVBoxLayout(group)
        
        # Currencies table
        currencies_label = QLabel("Latest Currencies:")
        layout.addWidget(currencies_label)
        
        self.currencies_table = QTableWidget(1, 4)
        self.currencies_table.setHorizontalHeaderLabels(["Oil", "Coins", "Gems", "Cubes"])
        self.currencies_table.setMaximumHeight(80)
        layout.addWidget(self.currencies_table)
        
        # Commissions table
        commissions_label = QLabel("Latest Commissions:")
        layout.addWidget(commissions_label)
        
        self.commissions_table = QTableWidget(0, 5)
        self.commissions_table.setHorizontalHeaderLabels(["Slot", "Name", "Rarity", "Time", "Status"])
        layout.addWidget(self.commissions_table)
        
        return group
    
    def create_llm_group(self) -> QGroupBox:
        """Create the LLM JSON view group."""
        group = QGroupBox("LLM Response")
        group.setCheckable(True)
        group.setChecked(False)  # Collapsed by default
        layout = QVBoxLayout(group)
        
        self.llm_text = QTextEdit()
        self.llm_text.setMaximumHeight(150)
        self.llm_text.setReadOnly(True)
        self.llm_text.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.llm_text)
        
        return group
    
    def init_timer(self) -> None:
        """Initialize the refresh timer."""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(500)  # 2 FPS refresh
    
    def load_configuration(self) -> None:
        """Load configuration and bootstrap components."""
        try:
            # Look for config file
            config_paths = [
                Path("config/app.yaml"),
                Path("~/.azlbot/config.yaml").expanduser()
            ]
            
            config_path = None
            for path in config_paths:
                if path.exists():
                    config_path = path
                    break
            
            if config_path:
                self.add_status(f"Loading config from: {config_path}")
                self.components = bootstrap_from_config(config_path)
            else:
                self.add_status("No config file found, using defaults")
                config = create_default_config()
                # Set reasonable defaults for UI mode
                config.emulator.adb_serial = "127.0.0.1:5555"
                self.components = bootstrap_from_config_object(config)
            
            self.add_status("Components initialized successfully")
            self.update_data_display()
            
        except Exception as e:
            self.add_status(f"Failed to initialize components: {e}")
            logger.error(f"Bootstrap failed: {e}")
    
    def start_task(self) -> None:
        """Start the selected task."""
        if not self.components or not self.components.get("planner"):
            self.add_status("Cannot start task: components not initialized")
            return
        
        task_name = self.task_combo.currentText()
        task = self.components["tasks"].get(task_name)
        
        if not task:
            self.add_status(f"Task not found: {task_name}")
            return
        
        # Start worker thread
        self.worker_thread = QThread()
        self.worker = TaskWorker(self.components["planner"], task)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker_thread.started.connect(self.worker.run_task)
        self.worker.finished.connect(self.task_finished)
        self.worker.progress.connect(self.add_status)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Start thread
        self.worker_thread.start()
        self.add_status(f"Started task: {task_name}")
    
    def stop_task(self) -> None:
        """Stop the running task."""
        if self.worker:
            self.worker.stop()
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(5000)  # Wait up to 5 seconds
        
        self.task_finished(False)
        self.add_status("Task stopped by user")
    
    def run_once(self) -> None:
        """Run the selected task once."""
        # For now, just alias to start_task
        # In a full implementation, this would be a single execution
        self.start_task()
    
    def task_finished(self, success: bool) -> None:
        """Handle task completion."""
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # Update data
        self.update_data_display()
        
        status = "completed successfully" if success else "failed"
        self.add_status(f"Task {status}")
    
    def refresh_display(self) -> None:
        """Refresh the frame display."""
        if not self.components:
            return
        
        try:
            # Capture frame
            capture = self.components.get("capture")
            if capture:
                frame = capture.grab()
                self.display_frame(frame)
                
        except Exception as e:
            # Silently ignore capture errors (expected when device not connected)
            pass
    
    def display_frame(self, frame: Frame) -> None:
        """Display a frame in the UI.
        
        Args:
            frame: Frame to display
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply overlays if enabled
            if self.overlay_checkbox.isChecked():
                rgb_image = self.overlay_renderer.render_overlays(rgb_image, {
                    "regions": self.components["config"].resolver.regions.__dict__ if self.components else {},
                    "last_tap": self.ui_state.last_tap_point
                })
            
            # Convert to QImage
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(qt_image)
            self.frame_label.setPixmap(pixmap)
            
            # Update frame info
            self.frame_info.setText(f"Frame: {w}x{h}, Active: {frame.active_rect}, Time: {frame.ts:.1f}")
            
        except Exception as e:
            logger.debug(f"Frame display error: {e}")
    
    def update_data_display(self) -> None:
        """Update the data tables."""
        if not self.components:
            return
        
        try:
            datastore = self.components.get("datastore")
            if not datastore:
                return
            
            # Update currencies
            latest_currency = datastore.get_latest_currencies()
            if latest_currency:
                self.currencies_table.setItem(0, 0, QTableWidgetItem(str(latest_currency.oil or "N/A")))
                self.currencies_table.setItem(0, 1, QTableWidgetItem(str(latest_currency.coins or "N/A")))
                self.currencies_table.setItem(0, 2, QTableWidgetItem(str(latest_currency.gems or "N/A")))
                self.currencies_table.setItem(0, 3, QTableWidgetItem(str(latest_currency.cubes or "N/A")))
            
            # Update commissions
            latest_commissions = datastore.get_latest_commissions()
            self.commissions_table.setRowCount(len(latest_commissions))
            
            for i, commission in enumerate(latest_commissions):
                self.commissions_table.setItem(i, 0, QTableWidgetItem(str(commission.slot_id)))
                self.commissions_table.setItem(i, 1, QTableWidgetItem(commission.name or ""))
                self.commissions_table.setItem(i, 2, QTableWidgetItem(commission.rarity or ""))
                
                # Format time
                if commission.time_remaining_s:
                    hours = commission.time_remaining_s // 3600
                    minutes = (commission.time_remaining_s % 3600) // 60
                    time_str = f"{hours:02d}:{minutes:02d}"
                else:
                    time_str = "N/A"
                
                self.commissions_table.setItem(i, 3, QTableWidgetItem(time_str))
                self.commissions_table.setItem(i, 4, QTableWidgetItem(commission.status or ""))
            
        except Exception as e:
            logger.debug(f"Data update error: {e}")
    
    def add_status(self, message: str) -> None:
        """Add a status message.
        
        Args:
            message: Status message to add
        """
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.status_text.append(formatted_message)
        self.status_bar.showMessage(message)
        
        # Keep only last 100 lines
        document = self.status_text.document()
        if document.blockCount() > 100:
            cursor = self.status_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, document.blockCount() - 100)
            cursor.removeSelectedText()


def main() -> None:
    """Main entry point for the UI application."""
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())