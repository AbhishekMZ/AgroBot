# hal/chassis.py
class ChassisController:
    def __init__(self, config, comm_module=None):
        """Initialize chassis controller.
        
        Args:
            config: Dict with motor settings
            comm_module: Communication module for Arduino (if used)
        """
        self.config = config
        self.comm = comm_module
        self.current_speed = 0
        self.is_moving = False
        
    def move_forward(self, speed=None):
        """Start forward movement.
        
        Args:
            speed: Speed value (0-100%)
        """
        if speed is None:
            speed = self.config['default_speed']
        
        if self.comm:
            # Send command to Arduino
            self.comm.send_command(f"MOVE:FWD,{speed}")
        else:
            # Direct GPIO control if not using Arduino
            # Use RPi.GPIO or similar library here
            print(f"Moving forward at {speed}%")
        
        self.is_moving = True
        self.current_speed = speed
    
    def stop(self):
        """Stop all movement."""
        if self.comm:
            self.comm.send_command("MOVE:STOP")
        else:
            # Direct GPIO control
            print("Stopping motors")
        
        self.is_moving = False
        self.current_speed = 0