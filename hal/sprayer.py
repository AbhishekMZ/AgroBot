# hal/sprayer.py
import time
import logging
import threading

class SprayerActuator:
    """Hardware abstraction layer for sprayer control.
    
    This class provides a consistent interface for controlling the sprayer,
    abstracting the underlying hardware implementation.
    """
    
    def __init__(self, config, comm_module=None):
        """Initialize sprayer control.
        
        Args:
            config: Dictionary with sprayer settings
                - default_duration: Default spray duration in seconds
                - min_duration: Minimum spray duration
                - max_duration: Maximum spray duration
                - pin: GPIO pin number for direct control
                - pulse_width: PWM pulse width for variable spray (0-100)
                - direct_control: Whether to use direct GPIO control
            comm_module: Communication module for external sprayer controller
        """
        self.logger = logging.getLogger("Sprayer")
        self.config = config
        self.comm = comm_module
        
        # Sprayer parameters
        self.default_duration = config.get('default_duration', 1.0)
        self.min_duration = config.get('min_duration', 0.1)
        self.max_duration = config.get('max_duration', 5.0)
        self.pulse_width = config.get('pulse_width', 100)  # Full power by default
        
        # State variables
        self.is_active = False
        self.spray_start_time = None
        self.spray_timer = None
        
        # Control method
        self.direct_control = config.get('direct_control', False)
        
        # Initialize hardware interfaces
        if self.direct_control:
            self._initialize_direct_control(config)
        else:
            self._validate_comm_module()
            
        self.logger.info("Sprayer actuator initialized")
    
    def _initialize_direct_control(self, config):
        """Initialize direct GPIO control for sprayer.
        
        Args:
            config: Configuration dictionary with pin information
        """
        try:
            # Only import GPIO libraries if needed
            import RPi.GPIO as GPIO
            
            # Configure GPIO
            GPIO.setmode(GPIO.BCM)
            
            # Setup pin for sprayer
            self.pin = config.get('pin')
            if self.pin is None:
                raise ValueError("Pin number must be specified for direct control")
                
            GPIO.setup(self.pin, GPIO.OUT)
            
            # Initialize PWM for variable spray control if needed
            self.pwm_control = config.get('pwm_control', False)
            if self.pwm_control:
                self.pwm = GPIO.PWM(self.pin, 100)  # 100 Hz frequency
                self.pwm.start(0)  # Start with duty cycle 0 (off)
            else:
                # Ensure sprayer is off
                GPIO.output(self.pin, GPIO.LOW)
                
            self.logger.info(f"Direct GPIO control initialized on pin {self.pin}")
            
        except ImportError:
            self.logger.warning("RPi.GPIO not available, using mock GPIO")
            self.direct_control = False
        except Exception as e:
            self.logger.error(f"Error initializing GPIO: {e}")
            self.direct_control = False
    
    def _validate_comm_module(self):
        """Validate communication module for external sprayer control."""
        if self.comm is None:
            self.logger.warning("No communication module provided for sprayer control")
            return False
            
        # Check if comm module has required methods
        required_methods = ['send_command', 'read_response']
        for method in required_methods:
            if not hasattr(self.comm, method):
                self.logger.error(f"Communication module missing required method: {method}")
                return False
                
        return True
    
    def _constrain_duration(self, duration):
        """Constrain duration to valid range.
        
        Args:
            duration: Raw duration value in seconds
            
        Returns:
            float: Constrained duration value
        """
        if duration is None:
            return self.default_duration
            
        return max(self.min_duration, min(float(duration), self.max_duration))
    
    def _direct_control_spray(self, active, pulse_width=None):
        """Directly control sprayer via GPIO.
        
        Args:
            active: Whether sprayer should be active
            pulse_width: PWM duty cycle for variable spray (0-100)
        """
        try:
            import RPi.GPIO as GPIO
            
            if self.pwm_control:
                if active:
                    # Set PWM duty cycle
                    pw = pulse_width if pulse_width is not None else self.pulse_width
                    self.pwm.ChangeDutyCycle(pw)
                else:
                    # Turn off
                    self.pwm.ChangeDutyCycle(0)
            else:
                # Simple on/off control
                GPIO.output(self.pin, GPIO.HIGH if active else GPIO.LOW)
                
        except Exception as e:
            self.logger.error(f"Error controlling sprayer: {e}")
    
    def _auto_deactivate_callback(self):
        """Timer callback to automatically deactivate sprayer."""
        self.deactivate()
    
    def activate(self, duration=None, pulse_width=None):
        """Activate sprayer for specified duration.
        
        Args:
            duration: Spray duration in seconds (defaults to config value)
            pulse_width: PWM duty cycle for variable spray (0-100)
        """
        # Constrain duration
        duration = self._constrain_duration(duration)
        
        # Save pulse width if provided
        if pulse_width is not None:
            self.pulse_width = max(0, min(100, pulse_width))
        
        # Cancel any existing timer
        if self.spray_timer:
            self.spray_timer.cancel()
            self.spray_timer = None
        
        # Activate sprayer
        if self.direct_control:
            self._direct_control_spray(True, self.pulse_width)
        elif self.comm:
            # Send command to external controller
            self.comm.send_command(f"SPRAY:ON,{duration}")
        else:
            self.logger.info(f"MOCK: Activating sprayer for {duration:.1f}s at {self.pulse_width}% power")
        
        # Update state
        self.is_active = True
        self.spray_start_time = time.time()
        
        # Set timer for auto-deactivation
        self.spray_timer = threading.Timer(duration, self._auto_deactivate_callback)
        self.spray_timer.daemon = True
        self.spray_timer.start()
        
        self.logger.info(f"Sprayer activated for {duration:.1f}s")
        
        return duration
    
    def deactivate(self):
        """Immediately stop spraying."""
        # Cancel any existing timer
        if self.spray_timer:
            self.spray_timer.cancel()
            self.spray_timer = None
        
        # Only send deactivation command if currently active
        if self.is_active:
            if self.direct_control:
                self._direct_control_spray(False)
            elif self.comm:
                self.comm.send_command("SPRAY:OFF")
            else:
                self.logger.info("MOCK: Deactivating sprayer")
            
            # Calculate spray duration if active
            if self.spray_start_time:
                spray_duration = time.time() - self.spray_start_time
                self.logger.info(f"Sprayer deactivated after {spray_duration:.1f}s")
            else:
                self.logger.info("Sprayer deactivated")
                
            # Update state
            self.is_active = False
            self.spray_start_time = None
    
    def get_status(self):
        """Get current sprayer status.
        
        Returns:
            dict: Current sprayer status
        """
        status = {
            "is_active": self.is_active,
            "pulse_width": self.pulse_width
        }
        
        # Add spray duration if active
        if self.is_active and self.spray_start_time:
            status["spray_duration"] = time.time() - self.spray_start_time
            
        return status