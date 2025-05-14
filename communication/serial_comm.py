# communication/serial_comm.py
import serial
import time
import threading
import queue
import logging

class SerialCommunicator:
    """Communication module for serial communication with external controllers.
    
    This class provides methods for sending commands and receiving responses
    via serial connection, with optional asynchronous reading.
    """
    
    def __init__(self, config):
        """Initialize serial communication.
        
        Args:
            config: Dictionary with serial settings
                - port: Serial port name (e.g., '/dev/ttyACM0', 'COM3')
                - baud_rate: Baud rate (e.g., 9600, 115200)
                - timeout: Read timeout in seconds
                - async_read: Whether to use asynchronous reading
                - buffer_size: Response buffer size for async reading
        """
        self.logger = logging.getLogger("SerialComm")
        self.config = config
        
        # Serial parameters
        self.port = config.get('port')
        self.baud_rate = config.get('baud_rate', 115200)
        self.timeout = config.get('timeout', 1.0)
        
        # Threading parameters
        self.async_read = config.get('async_read', True)
        self.response_buffer = queue.Queue(maxsize=config.get('buffer_size', 100))
        
        # Initialize variables
        self.serial = None
        self.connected = False
        self.reading_thread = None
        self.running = False
        
        # Debug mode for development without hardware
        self.debug_mode = config.get('debug_mode', False)
        
        # Connect if port is specified
        if self.port:
            self.connect()
    
    def connect(self):
        """Establish serial connection.
        
        Returns:
            bool: Success status
        """
        if self.debug_mode:
            self.logger.info(f"DEBUG MODE: Simulating connection to {self.port}")
            self.connected = True
            
            # Start mock reading thread if async_read
            if self.async_read:
                self._start_reading_thread()
                
            return True
            
        try:
            self.logger.info(f"Connecting to {self.port} at {self.baud_rate} baud")
            
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            
            # Wait for Arduino reset if using Arduino
            time.sleep(2)
            
            self.connected = True
            self.logger.info(f"Connected to {self.port}")
            
            # Start reading thread if async_read
            if self.async_read:
                self._start_reading_thread()
                
            return True
            
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to {self.port}: {e}")
            self.connected = False
            return False
    
    def _start_reading_thread(self):
        """Start background thread for asynchronous reading."""
        if self.reading_thread is not None and self.reading_thread.is_alive():
            return
            
        self.running = True
        self.reading_thread = threading.Thread(target=self._reading_loop)
        self.reading_thread.daemon = True
        self.reading_thread.start()
        self.logger.debug("Async reading thread started")
    
    def _stop_reading_thread(self):
        """Stop the background reading thread."""
        self.running = False
        if self.reading_thread:
            self.reading_thread.join(timeout=1.0)
            self.reading_thread = None
    
    def _reading_loop(self):
        """Continuous reading loop for background thread."""
        while self.running:
            try:
                if self.debug_mode:
                    # Simulate random responses in debug mode
                    time.sleep(0.5)
                    if self.last_command.startswith("MOVE:"):
                        response = f"OK:{self.last_command[5:]}"
                    elif self.last_command.startswith("SPRAY:"):
                        response = f"OK:{self.last_command[6:]}"
                    else:
                        response = "OK:DEBUG"
                        
                    self.response_buffer.put(response)
                    continue
                
                # Check if data is available
                if not self.connected or not self.serial or not self.serial.in_waiting:
                    time.sleep(0.01)  # Short sleep to prevent CPU hogging
                    continue
                
                # Read a line (until newline)
                line = self.serial.readline().decode('utf-8').strip()
                
                if line:
                    self.logger.debug(f"Received: {line}")
                    
                    # Add to response buffer
                    if self.response_buffer.full():
                        try:
                            # Remove oldest message if buffer is full
                            self.response_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.response_buffer.put(line)
            
            except Exception as e:
                self.logger.error(f"Error in reading thread: {e}")
                time.sleep(0.1)  # Avoid tight error loop
    
    def send_command(self, command):
        """Send command string to the connected device.
        
        Args:
            command: Command string to send
            
        Returns:
            bool: Success status
        """
        if self.debug_mode:
            self.logger.info(f"DEBUG MODE: Sending: {command}")
            self.last_command = command  # Store for mock responses
            return True
            
        if not self.connected:
            self.logger.warning("Cannot send command: not connected")
            return False
            
        try:
            # Add newline terminator if needed
            if not command.endswith('\n'):
                command += '\n'
                
            # Send command
            self.serial.write(command.encode('utf-8'))
            self.logger.debug(f"Sent: {command.strip()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return False
    
    def read_response(self, timeout=1.0):
        """Read response from the connected device.
        
        Args:
            timeout: Read timeout in seconds
            
        Returns:
            str: Response string or None if timeout
        """
        if self.debug_mode:
            # In debug mode, returns a mock response related to the last command
            try:
                return self.response_buffer.get(timeout=timeout)
            except queue.Empty:
                return None
                
        if not self.connected:
            self.logger.warning("Cannot read response: not connected")
            return None
            
        # If using async reading, get from buffer
        if self.async_read:
            try:
                return self.response_buffer.get(timeout=timeout)
            except queue.Empty:
                return None
        
        # Direct synchronous reading
        try:
            start_time = time.time()
            response = ""
            
            while (time.time() - start_time) < timeout:
                if self.serial.in_waiting > 0:
                    char = self.serial.read().decode('utf-8')
                    if char == '\n':
                        break
                    response += char
                    
                # Short sleep to prevent CPU hogging
                time.sleep(0.001)
                
            return response.strip() if response else None
            
        except Exception as e:
            self.logger.error(f"Error reading response: {e}")
            return None
    
    def flush(self):
        """Flush input and output buffers."""
        if self.debug_mode:
            # Clear mock buffer
            while not self.response_buffer.empty():
                try:
                    self.response_buffer.get_nowait()
                except queue.Empty:
                    break
            return
            
        if not self.connected or not self.serial:
            return
            
        try:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
        except Exception as e:
            self.logger.error(f"Error flushing buffers: {e}")
    
    def close(self):
        """Close the serial connection."""
        self.logger.info("Closing serial connection")
        
        # Stop reading thread
        if self.async_read:
            self._stop_reading_thread()
        
        # Close serial port
        if self.serial and not self.debug_mode:
            try:
                self.serial.close()
            except Exception as e:
                self.logger.error(f"Error closing serial port: {e}")
        
        self.connected = False
        self.serial = None