# communication/serial_comm.py
import serial
import time

class SerialCommunicator:
    def __init__(self, config):
        """Initialize serial communication.
        
        Args:
            config: Dict with serial settings (port, baud rate)
        """
        self.port = config['port']
        self.baud_rate = config['baud_rate']
        self.timeout = config.get('timeout', 1.0)
        self.serial = None
        self.connected = False
        
    def connect(self):
        """Establish serial connection."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            time.sleep(2)  # Allow Arduino to reset
            self.connected = True
            print(f"Connected to {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            self.connected = False
            return False
    
    def send_command(self, command):
        """Send command string to Arduino.
        
        Args:
            command: Command string (e.g., "MOVE:FWD,50")
        
        Returns:
            bool: Success status
        """
        if not self.connected:
            print("Not connected")
            return False
            
        try:
            # Add newline as command terminator
            cmd = command + "\n"
            self.serial.write(cmd.encode())
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def read_response(self, timeout=1.0):
        """Read response from Arduino.
        
        Args:
            timeout: Read timeout in seconds
        
        Returns:
            str: Response string or None
        """
        if not self.connected:
            return None
            
        try:
            # Wait for response until newline
            start_time = time.time()
            response = ""
            
            while (time.time() - start_time) < timeout:
                if self.serial.in_waiting > 0:
                    char = self.serial.read().decode()
                    if char == '\n':
                        break
                    response += char
                time.sleep(0.01)
                
            return response.strip() if response else None
        except Exception as e:
            print(f"Read error: {e}")
            return None
    
    def close(self):
        """Close the serial connection."""
        if self.serial and self.connected:
            self.serial.close()
            self.connected = False