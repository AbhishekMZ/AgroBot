# app/controller.py
import time
import threading
import cv2
import numpy as np
from utils.config import load_config
from hal.camera import PiCamera
from hal.chassis import ChassisController
from hal.sprayer import SprayerActuator
from communication.serial_comm import SerialCommunicator
from core.inference import AIInferenceModule
from core.decision import DecisionModule
import logging

class AgroBotController:
    def __init__(self, config_path="config/settings.yaml"):
        """Initialize the AgroBot controller.
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("agrobot.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AgroBot")
        
        # Load configuration
        self.logger.info("Loading configuration")
        self.config = load_config(config_path)
        
        # Initialize communication if needed
        self.comm = None
        if self.config.get('communication', {}).get('port'):
            self.logger.info("Initializing communication")
            self.comm = SerialCommunicator(self.config['communication'])
            self.comm.connect()
        
        # Initialize hardware components
        self.logger.info("Initializing hardware components")
        self.camera = PiCamera(self.config['camera'])
        self.chassis = ChassisController(self.config['chassis'], self.comm)
        self.sprayer = SprayerActuator(self.config['sprayer'], self.comm)
        
        # Initialize AI components
        self.logger.info("Initializing AI components")
        self.inference = AIInferenceModule(self.config['model'])
        self.decision = DecisionModule(self.config['app'])
        
        # State variables
        self.running = False
        self.detection_thread = None
        
        self.logger.info("Initialization complete")
    
    def start(self):
        """Start the AgroBot."""
        if self.running:
            self.logger.warning("Already running")
            return
            
        self.running = True
        self.logger.info("Starting AgroBot")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Start moving forward
        self.chassis.move_forward()
    
    def stop(self):
        """Stop the AgroBot."""
        if not self.running:
            return
            
        self.running = False
        self.logger.info("Stopping AgroBot")
        
        # Stop movement
        self.chassis.stop()
        
        # Wait for detection thread to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
    
    def _detection_loop(self):
        """Main detection and control loop."""
        self.logger.info("Detection loop started")
        
        while self.running:
            try:
                # Capture frame
                frame = self.camera.capture_frame()
                
                # Run inference
                detection_result = self.inference.detect(frame)
                
                # Log inference time
                self.logger.debug(f"Inference time: {detection_result['time_ms']:.1f} ms")
                
                # Make decision
                decision = self.decision.interpret_detection(detection_result)
                
                # Take action based on decision
                if decision['action'] == 'SPRAY':
                    self.logger.info(f"Weed detected! Confidence: {decision['confidence']:.2f}")
                    
                    # Stop movement
                    self.chassis.stop()
                    
                    # Small delay to ensure complete stop
                    time.sleep(0.5)
                    
                    # Activate sprayer
                    self.sprayer.activate()
                    
                    # Wait for spray to complete
                    spray_duration = self.config['sprayer']['default_duration']
                    time.sleep(spray_duration + 0.5)  # Add small buffer
                    
                    # Resume movement
                    self.chassis.move_forward()
                
                # Wait for next detection cycle
                time.sleep(self.config['app']['detection_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {str(e)}")
                time.sleep(1.0)  # Avoid rapid error loops
    
    def cleanup(self):
        """Release all resources."""
        self.logger.info("Cleaning up resources")
        
        # Stop detection loop if running
        if self.running:
            self.stop()
        
        # Release hardware resources
        self.camera.close()
        
        # Close communication
        if self.comm:
            self.comm.close()