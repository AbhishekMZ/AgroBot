# hal/sprayer.py
class SprayerActuator:
    def __init__(self, config, comm_module=None):
        """Initialize sprayer control.
        
        Args:
            config: Dict with sprayer settings (duration, etc.)
            comm_module: Communication module for Arduino (if used)
        """
        self.config = config
        self.comm = comm_module
        self.is_active = False
        
    def activate(self, duration=None):
        """Activate sprayer for specified duration.
        
        Args:
            duration: Spray duration in seconds (defaults to config value)
        """
        if duration is None:
            duration = self.config['default_duration']
            
        if self.comm:
            self.comm.send_command(f"SPRAY:ON,{duration}")
        else:
            # Direct GPIO control
            print(f"Spraying for {duration} seconds")
            
        self.is_active = True
        
        # If using direct control, you'd need threading here
        # to turn off after duration expires
        
    def deactivate(self):
        """Immediately stop spraying."""
        if self.comm:
            self.comm.send_command("SPRAY:OFF")
        else:
            # Direct GPIO control
            print("Stopping sprayer")
            
        self.is_active = False