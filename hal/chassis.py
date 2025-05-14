# hal/chassis.py
import time
import logging
import threading

class ChassisController:
    """Hardware abstraction layer for rover chassis control.
    
    This class provides a consistent interface for controlling the rover's movement,
    abstracting the underlying hardware implementation (direct GPIO, Arduino, etc.)
    """
    
    def __init__(self, config, comm_module=None):
        """Initialize chassis controller.
        
        Args:
            config: Dictionary with chassis configuration
                - default_speed: Default speed value (0-100%)
                - max_speed: Maximum allowed speed (0-100%)
                - turn_factor: Factor to reduce inner wheel speed during turns
                - motor_type: Type of motors ('dc', 'servo', etc.)
                - direct_control: Whether to use direct GPIO control
                - left_pins: List of pins for left motor if using direct control
                - right_pins: List of pins for right motor if using direct control
            comm_module: Communication module for external motor controller
        """
        self.logger = logging.getLogger("Chassis")
        self.config = config
        self.comm = comm_module
        
        # Movement parameters
        self.default_speed = config.get('default_speed', 50)
        self.max_speed = config.get('max_speed', 100)
        self.turn_factor = config.get('turn_factor', 0.6)
        
        # State variables
        self.current_speed = 0
        self.is_moving = False
        self.direction = "stop"  # stop, forward, backward, left, right
        
        # Control method
        self.direct_control = config.get('direct_control', False)
        self.motor_type = config.get('motor_type', 'dc')
        
        # Initialize hardware interfaces
        if self.direct_control:
            self._initialize_direct_control(config)
        else:
            self._validate_comm_module()
            
        self.logger.info("Chassis controller initialized")
    
    def _initialize_direct_control(self, config):
        """Initialize direct GPIO control for motors.
        
        Args:
            config: Configuration dictionary with pin information
        """
        try:
            # Only import GPIO libraries if needed
            import RPi.GPIO as GPIO
            
            # Configure GPIO
            GPIO.setmode(GPIO.BCM)
            
            # Setup pins for left motor
            self.left_pins = config.get('left_pins', [])
            for pin in self.left_pins:
                GPIO.setup(pin, GPIO.OUT)
                
            # Setup pins for right motor
            self.right_pins = config.get('right_pins', [])
            for pin in self.right_pins:
                GPIO.setup(pin, GPIO.OUT)
                
            # Setup PWM if needed
            if len(self.left_pins) >= 3:  # Assuming [EN, IN1, IN2] format
                self.left_pwm = GPIO.PWM(self.left_pins[0], 100)
                self.left_pwm.start(0)
                
            if len(self.right_pins) >= 3:  # Assuming [EN, IN1, IN2] format
                self.right_pwm = GPIO.PWM(self.right_pins[0], 100)
                self.right_pwm.start(0)
                
            self.logger.info("Direct GPIO control initialized")
            
        except ImportError:
            self.logger.warning("RPi.GPIO not available, using mock GPIO")
            self.direct_control = False
        except Exception as e:
            self.logger.error(f"Error initializing GPIO: {e}")
            self.direct_control = False
    
    def _validate_comm_module(self):
        """Validate communication module for external motor control."""
        if self.comm is None:
            self.logger.warning("No communication module provided for chassis control")
            return False
            
        # Check if comm module has required methods
        required_methods = ['send_command', 'read_response']
        for method in required_methods:
            if not hasattr(self.comm, method):
                self.logger.error(f"Communication module missing required method: {method}")
                return False
                
        # Test communication
        try:
            self.comm.send_command("PING")
            response = self.comm.read_response(timeout=1.0)
            if response:
                self.logger.info(f"Communication test successful: {response}")
            else:
                self.logger.warning("No response from communication module")
        except Exception as e:
            self.logger.error(f"Error testing communication: {e}")
            
        return True
    
    def _constrain_speed(self, speed):
        """Constrain speed to valid range.
        
        Args:
            speed: Raw speed value
            
        Returns:
            int: Constrained speed value (0-100)
        """
        if speed is None:
            return self.default_speed
            
        return max(0, min(int(speed), self.max_speed))
    
    def _direct_control_motors(self, left_speed, right_speed):
        """Directly control motors via GPIO.
        
        Args:
            left_speed: Speed for left motor (-100 to 100)
            right_speed: Speed for right motor (-100 to 100)
        """
        try:
            import RPi.GPIO as GPIO
            
            # Left motor
            left_direction = left_speed >= 0
            left_power = abs(left_speed)
            
            # Right motor
            right_direction = right_speed >= 0
            right_power = abs(right_speed)
            
            # Set direction pins for left motor
            if len(self.left_pins) >= 3:  # Using [EN, IN1, IN2] format
                GPIO.output(self.left_pins[1], left_direction)
                GPIO.output(self.left_pins[2], not left_direction)
                self.left_pwm.ChangeDutyCycle(left_power)
            elif len(self.left_pins) >= 2:  # Using [FWD, REV] format with PWM on both
                if left_direction:
                    GPIO.output(self.left_pins[1], GPIO.LOW)
                    GPIO.output(self.left_pins[0], GPIO.HIGH if left_power > 0 else GPIO.LOW)
                else:
                    GPIO.output(self.left_pins[0], GPIO.LOW)
                    GPIO.output(self.left_pins[1], GPIO.HIGH if left_power > 0 else GPIO.LOW)
            
            # Set direction pins for right motor
            if len(self.right_pins) >= 3:  # Using [EN, IN1, IN2] format
                GPIO.output(self.right_pins[1], right_direction)
                GPIO.output(self.right_pins[2], not right_direction)
                self.right_pwm.ChangeDutyCycle(right_power)
            elif len(self.right_pins) >= 2:  # Using [FWD, REV] format
                if right_direction:
                    GPIO.output(self.right_pins[1], GPIO.LOW)
                    GPIO.output(self.right_pins[0], GPIO.HIGH if right_power > 0 else GPIO.LOW)
                else:
                    GPIO.output(self.right_pins[0], GPIO.LOW)
                    GPIO.output(self.right_pins[1], GPIO.HIGH if right_power > 0 else GPIO.LOW)
                    
        except Exception as e:
            self.logger.error(f"Error controlling motors: {e}")
    
    def move_forward(self, speed=None):
        """Start forward movement.
        
        Args:
            speed: Speed value (0-100%)
        """
        speed = self._constrain_speed(speed)
        
        if self.direct_control:
            self._direct_control_motors(speed, speed)
        elif self.comm:
            # Send command to external controller
            self.comm.send_command(f"MOVE:FWD,{speed}")
        else:
            self.logger.info(f"MOCK: Moving forward at {speed}%")
        
        self.is_moving = True
        self.current_speed = speed
        self.direction = "forward"
        
        self.logger.info(f"Moving forward at {speed}%")
    
    def move_backward(self, speed=None):
        """Start backward movement.
        
        Args:
            speed: Speed value (0-100%)
        """
        speed = self._constrain_speed(speed)
        
        if self.direct_control:
            self._direct_control_motors(-speed, -speed)
        elif self.comm:
            # Send command to external controller
            self.comm.send_command(f"MOVE:REV,{speed}")
        else:
            self.logger.info(f"MOCK: Moving backward at {speed}%")
        
        self.is_moving = True
        self.current_speed = speed
        self.direction = "backward"
        
        self.logger.info(f"Moving backward at {speed}%")
    
    def turn_left(self, speed=None, turn_radius=1.0):
        """Turn left while moving.
        
        Args:
            speed: Forward speed value (0-100%)
            turn_radius: Adjusts turning sharpness (0.0-1.0)
                0.0 = pivot on spot (right wheel full, left wheel stopped)
                1.0 = gentle turn (right wheel full, left wheel at turn_factor)
        """
        speed = self._constrain_speed(speed)
        turn_radius = max(0.0, min(1.0, turn_radius))
        
        # Calculate wheel speeds
        right_speed = speed
        left_speed = speed * (self.turn_factor * turn_radius)
        
        if self.direct_control:
            self._direct_control_motors(left_speed, right_speed)
        elif self.comm:
            # Send command to external controller
            self.comm.send_command(f"MOVE:LEFT,{speed},{turn_radius:.1f}")
        else:
            self.logger.info(f"MOCK: Turning left at {speed}% (L:{left_speed:.0f}%, R:{right_speed:.0f}%)")
        
        self.is_moving = True
        self.current_speed = speed
        self.direction = "left"
        
        self.logger.info(f"Turning left at {speed}%")
    
    def turn_right(self, speed=None, turn_radius=1.0):
        """Turn right while moving.
        
        Args:
            speed: Forward speed value (0-100%)
            turn_radius: Adjusts turning sharpness (0.0-1.0)
                0.0 = pivot on spot (left wheel full, right wheel stopped)
                1.0 = gentle turn (left wheel full, right wheel at turn_factor)
        """
        speed = self._constrain_speed(speed)
        turn_radius = max(0.0, min(1.0, turn_radius))
        
        # Calculate wheel speeds
        left_speed = speed
        right_speed = speed * (self.turn_factor * turn_radius)
        
        if self.direct_control:
            self._direct_control_motors(left_speed, right_speed)
        elif self.comm:
            # Send command to external controller
            self.comm.send_command(f"MOVE:RIGHT,{speed},{turn_radius:.1f}")
        else:
            self.logger.info(f"MOCK: Turning right at {speed}% (L:{left_speed:.0f}%, R:{right_speed:.0f}%)")
        
        self.is_moving = True
        self.current_speed = speed
        self.direction = "right"
        
        self.logger.info(f"Turning right at {speed}%")
    
    def stop(self):
        """Stop all movement."""
        if self.direct_control:
            self._direct_control_motors(0, 0)
        elif self.comm:
            self.comm.send_command("MOVE:STOP")
        else:
            self.logger.info("MOCK: Stopping motors")
        
        self.is_moving = False
        self.current_speed = 0
        self.direction = "stop"
        
        self.logger.info("Motors stopped")
    
    def get_status(self):
        """Get current chassis status.
        
        Returns:
            dict: Current chassis status
        """
        return {
            "is_moving": self.is_moving,
            "speed": self.current_speed,
            "direction": self.direction
        }