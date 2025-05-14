# hal/camera.py
import time
import cv2
import numpy as np
import logging
import threading
from queue import Queue, Empty

class CameraInterface:
    """Hardware abstraction layer for camera.
    
    This class provides a consistent interface for camera operations,
    abstracting the underlying hardware implementation.
    """
    
    def __init__(self, config):
        """Initialize camera interface.
        
        Args:
            config: Dictionary containing camera configuration
                - width: Image width
                - height: Image height
                - fps: Frame rate
                - rotation: Image rotation (0, 90, 180, 270)
                - buffer_size: Frame buffer size
                - use_threading: Whether to use threading for capture
        """
        self.logger = logging.getLogger("Camera")
        self.config = config
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps = config.get('fps', 30)
        self.rotation = config.get('rotation', 0)
        self.use_threading = config.get('use_threading', True)
        
        # Frame buffer for threaded capture
        self.buffer_size = config.get('buffer_size', 1)
        self.frame_buffer = Queue(maxsize=self.buffer_size)
        
        # Initialize camera
        self.camera = None
        self.is_running = False
        self.capture_thread = None
        
        # Attempt to initialize appropriate camera implementation
        self._initialize_camera()
        
    def _initialize_camera(self):
        """Initialize appropriate camera based on available hardware.
        
        This method tries to detect available camera hardware and initialize
        the appropriate implementation.
        """
        try:
            # Try to import picamera2 for Raspberry Pi
            from picamera2 import Picamera2
            
            self.logger.info("Using PiCamera2 implementation")
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height), 
                      "format": "RGB888"}, 
                controls={"FrameRate": self.fps}
            )
            self.camera.configure(config)
            
            # Set rotation if needed
            if self.rotation:
                self.camera.rotation = self.rotation
                
            self.camera.start()
            time.sleep(2)  # Allow camera to stabilize
            
            self.camera_type = "picamera2"
            self.logger.info(f"Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
            
        except ImportError:
            # Fallback to OpenCV camera (webcam)
            self.logger.info("PiCamera2 not available, trying OpenCV camera")
            
            try:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                
                # Check if camera opened successfully
                if not self.camera.isOpened():
                    raise IOError("Failed to open OpenCV camera")
                    
                # Read a test frame
                ret, _ = self.camera.read()
                if not ret:
                    raise IOError("Failed to capture frame from OpenCV camera")
                
                self.camera_type = "opencv"
                self.logger.info(f"OpenCV camera initialized: {self.width}x{self.height}")
                
            except Exception as e:
                # Final fallback: create a dummy camera that returns test patterns
                self.logger.warning(f"Failed to initialize real camera: {e}")
                self.logger.info("Using dummy camera implementation")
                self.camera = None
                self.camera_type = "dummy"
                
        # Start threaded capture if enabled
        if self.use_threading and self.camera_type != "dummy":
            self._start_capture_thread()
    
    def _start_capture_thread(self):
        """Start background thread for continuous frame capture."""
        if self.capture_thread is not None and self.capture_thread.is_alive():
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        self.logger.info("Capture thread started")
    
    def _stop_capture_thread(self):
        """Stop the background capture thread."""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
        
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except Empty:
                break
    
    def _capture_loop(self):
        """Continuous frame capture loop for threaded operation."""
        while self.is_running:
            try:
                frame = self._capture_frame_internal()
                
                # If buffer is full, remove oldest frame
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                    except Empty:
                        pass
                
                self.frame_buffer.put(frame)
                
                # Short sleep to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in capture thread: {e}")
                time.sleep(0.1)  # Avoid tight error loop
    
    def _capture_frame_internal(self):
        """Internal method to capture a frame from the current camera."""
        if self.camera_type == "picamera2":
            # Capture from PiCamera2
            return self.camera.capture_array("main")
            
        elif self.camera_type == "opencv":
            # Capture from OpenCV camera
            ret, frame = self.camera.read()
            if not ret:
                raise IOError("Failed to capture frame from OpenCV camera")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply rotation if needed
            if self.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            return frame
            
        else:  # Dummy camera
            # Create a test pattern
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw some elements for a test pattern
            # Red, green and blue rectangles
            frame[20:120, 20:120] = [255, 0, 0]  # Red
            frame[20:120, 140:240] = [0, 255, 0]  # Green
            frame[20:120, 260:360] = [0, 0, 255]  # Blue
            
            # Add a timestamp text
            timestamp = time.strftime("%H:%M:%S")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Dummy Camera {timestamp}", (10, self.height - 30), 
                      font, 0.7, (255, 255, 255), 2)
            
            return frame
    
    def capture_frame(self):
        """Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray: RGB image array
        """
        start_time = time.time()
        
        # If using threaded capture, get frame from buffer
        if self.use_threading and self.camera_type != "dummy":
            try:
                frame = self.frame_buffer.get(timeout=1.0)
                capture_time = time.time() - start_time
                if capture_time > 0.1:  # Log if getting frame took too long
                    self.logger.warning(f"Slow frame retrieval: {capture_time:.3f}s")
                return frame
            except Empty:
                self.logger.warning("Frame buffer empty, capturing directly")
                # Fall back to direct capture
        
        # Direct capture
        try:
            frame = self._capture_frame_internal()
            return frame
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            # Return black frame on error
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def close(self):
        """Release camera resources."""
        self.logger.info("Closing camera")
        
        # Stop threaded capture
        if self.use_threading:
            self._stop_capture_thread()
        
        # Release camera resources
        if self.camera_type == "picamera2":
            if self.camera:
                self.camera.close()
        elif self.camera_type == "opencv":
            if self.camera:
                self.camera.release()
                
        self.camera = None