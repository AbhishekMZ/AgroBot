# hal/camera.py
class PiCamera:
    def __init__(self, config):
        """Initialize camera with configuration.
        
        Args:
            config: Dict containing camera settings (resolution, etc.)
        """
        self.resolution = (config['width'], config['height'])
        # Import here to avoid issues when running on non-Pi systems
        try:
            from picamera2 import Picamera2
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_preview_configuration(
                main={"size": self.resolution}))
            self.camera.start()
        except ImportError:
            print("PiCamera2 not available - using mock mode")
            self.camera = None
        
    def capture_frame(self):
        """Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray: Image as RGB numpy array
        """
        if self.camera:
            return self.camera.capture_array()
        else:
            # Return a test pattern when in development/mock mode
            import numpy as np
            return np.zeros((*self.resolution, 3), dtype=np.uint8)
    
    def close(self):
        """Release camera resources."""
        if self.camera:
            self.camera.close()