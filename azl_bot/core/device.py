"""Device abstraction for Android emulator control via ADB."""

import subprocess
import time
from typing import TypedDict

from loguru import logger


class DeviceInfo(TypedDict):
    """Device information returned by ADB."""
    width: int
    height: int
    density: int


class Device:
    """Android device interface using ADB."""
    
    def __init__(self, serial: str) -> None:
        """Initialize device with ADB serial.
        
        Args:
            serial: ADB device serial (e.g., "127.0.0.1:5555")
        """
        self.serial = serial
        self._info_cache: DeviceInfo | None = None
        
    def _adb(self, *args: str) -> bytes:
        """Execute ADB command and return output.
        
        Args:
            *args: ADB command arguments
            
        Returns:
            Command output as bytes
            
        Raises:
            subprocess.CalledProcessError: If ADB command fails
        """
        cmd = ["adb", "-s", self.serial] + list(args)
        logger.debug(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                check=True,
                timeout=10
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"ADB command timed out: {' '.join(cmd)}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"ADB command failed: {e.cmd}, stderr: {e.stderr.decode()}")
            raise
    
    def info(self) -> DeviceInfo:
        """Get device information.
        
        Returns:
            Device width, height and density
        """
        if self._info_cache is not None:
            return self._info_cache
            
        # Get window size
        output = self._adb("shell", "wm", "size").decode().strip()
        # Output format: "Physical size: 1920x1080" 
        size_line = output.split(":")[-1].strip()
        width_str, height_str = size_line.split("x")
        width, height = int(width_str), int(height_str)
        
        # Get density
        output = self._adb("shell", "wm", "density").decode().strip()
        # Output format: "Physical density: 480"
        density_line = output.split(":")[-1].strip()
        density = int(density_line)
        
        self._info_cache = DeviceInfo(
            width=width,
            height=height, 
            density=density
        )
        
        logger.info(f"Device info: {self._info_cache}")
        return self._info_cache
    
    def screencap_png(self) -> bytes:
        """Capture screen as PNG bytes.
        
        Returns:
            PNG image data
        """
        logger.debug("Capturing screen via ADB")
        return self._adb("exec-out", "screencap", "-p")
    
    def key_back(self) -> None:
        """Press the back key."""
        logger.debug("Pressing back key")
        self._adb("shell", "input", "keyevent", "KEYCODE_BACK")
        time.sleep(0.5)  # Brief delay for action to register
    
    def key_home(self) -> None:
        """Press the home key."""
        logger.debug("Pressing home key") 
        self._adb("shell", "input", "keyevent", "KEYCODE_HOME")
        time.sleep(0.5)
        
    def is_connected(self) -> bool:
        """Check if device is connected and responding.
        
        Returns:
            True if device responds to ADB commands
        """
        try:
            self._adb("shell", "echo", "test")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
            
    def reconnect(self) -> bool:
        """Attempt to reconnect to device.
        
        Returns:
            True if reconnection successful
        """
        try:
            logger.info(f"Attempting to reconnect to {self.serial}")
            subprocess.run(["adb", "disconnect", self.serial], capture_output=True)
            time.sleep(1)
            subprocess.run(["adb", "connect", self.serial], capture_output=True, check=True)
            time.sleep(2)
            return self.is_connected()
        except subprocess.CalledProcessError:
            logger.error(f"Failed to reconnect to {self.serial}")
            return False