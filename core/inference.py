# core/inference.py
import time
import numpy as np
import cv2
import logging
import os
import threading

class AIInferenceModule:
    """Core module for running AI model inference.
    
    This class handles loading TFLite models, preprocessing images,
    running inference, and interpreting results for weed detection.
    """
    
    def __init__(self, config):
        """Initialize inference module.
        
        Args:
            config: Dict with model settings
                - path: Path to TFLite model file
                - labels_path: Path to labels file
                - input_size: Model input dimensions [width, height]
                - threshold: Detection confidence threshold
                - input_mean: Input normalization mean (typically 127.5 or 0)
                - input_std: Input normalization std (typically 127.5 or 1.0)
                - enable_threading: Whether to use a separate thread for inference
                - continuous_mode: Whether to run continuous inference
        """
        self.logger = logging.getLogger("AIInference")
        self.config = config
        
        # Model parameters
        self.model_path = config.get('path')
        self.labels_path = config.get('labels_path')
        self.input_size = tuple(config.get('input_size', [224, 224]))
        self.threshold = config.get('threshold', 0.5)
        
        # Normalization parameters
        self.input_mean = config.get('input_mean', 127.5)
        self.input_std = config.get('input_std', 127.5)
        
        # Threading options
        self.enable_threading = config.get('enable_threading', False)
        self.continuous_mode = config.get('continuous_mode', False)
        
        # State variables
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        self.is_quantized = False
        self.inference_thread = None
        self.running = False
        self.last_result = None
        self.result_lock = threading.Lock()
        
        # Initialize model
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Load TFLite model and get input/output details."""
        if not os.path.exists(self.model_path):
            self.logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Import TFLite interpreter
            import tensorflow as tf
            
            # Load the TFLite model
            self.logger.info(f"Loading model: {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Check if model is quantized (INT8)
            self.is_quantized = self.input_details[0]['dtype'] == np.int8
            
            # Log model details
            input_shape = self.input_details[0]['shape']
            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Input shape: {input_shape}")
            self.logger.info(f"Input type: {self.input_details[0]['dtype']}")
            self.logger.info(f"Quantized: {self.is_quantized}")
            
            # Get quantization parameters if model is quantized
            if self.is_quantized:
                self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
                self.logger.info(f"Input scale: {self.input_scale}, zero point: {self.input_zero_point}")
                
        except ImportError:
            self.logger.error("TensorFlow Lite not available")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _load_labels(self):
        """Load labels file if available."""
        if not self.labels_path or not os.path.exists(self.labels_path):
            self.logger.warning(f"Labels file not found: {self.labels_path}")
            return
            
        try:
            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
                
            self.logger.info(f"Loaded {len(self.labels)} labels")
            
        except Exception as e:
            self.logger.error(f"Error loading labels: {e}")
    
    def _preprocess_image(self, image):
        """Preprocess image for model input.
        
        Args:
            image: RGB numpy array
            
        Returns:
            preprocessed_image: Resized and normalized numpy array
        """
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Convert to float and normalize
        if self.is_quantized:
            # For INT8 quantized model
            normalized = ((resized - self.input_mean) / self.input_std) / self.input_scale + self.input_zero_point
            normalized = normalized.astype(np.int8)
        else:
            # For float model
            normalized = (resized - self.input_mean) / self.input_std
            normalized = normalized.astype(np.float32)
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def _parse_output(self, output_data):
        """Parse model output based on model type.
        
        Args:
            output_data: Raw model output tensors
            
        Returns:
            dict: Structured detection results
        """
        # Determine model type based on output shape
        if len(self.output_details) >= 3:
            # Likely object detection model (SSD, EfficientDet, etc.)
            return self._parse_detection_output(output_data)
        else:
            # Likely classification model
            return self._parse_classification_output(output_data)
    
    def _parse_classification_output(self, output_data):
        """Parse classification model output.
        
        Args:
            output_data: Raw model output tensor
            
        Returns:
            dict: Classification results
        """
        # Get scores from output tensor
        scores = output_data[0]
        
        # Get top class index and score
        top_idx = np.argmax(scores)
        top_score = float(scores[top_idx])
        
        # Get class label
        if top_idx < len(self.labels):
            label = self.labels[top_idx]
        else:
            label = f"class_{top_idx}"
        
        return {
            "type": "classification",
            "class_id": int(top_idx),
            "label": label,
            "score": top_score,
            "all_scores": scores.tolist() if isinstance(scores, np.ndarray) else list(scores),
            "weed_detected": label.lower() == "weed" and top_score > self.threshold
        }
    
    def _parse_detection_output(self, output_data):
        """Parse object detection model output.
        
        Args:
            output_data: Raw model output tensors
            
        Returns:
            dict: Detection results
        """
        # Different models have different output formats
        # This handles common TFLite detection model formats
        
        # Try to identify output tensor order based on shapes
        boxes_idx = None
        classes_idx = None
        scores_idx = None
        
        for i, output in enumerate(self.output_details):
            # Look at output shape to guess its purpose
            shape = output['shape']
            
            if len(shape) == 3 and shape[1] == 4:
                # [batch, 4, num_detections] or similar - likely boxes
                boxes_idx = i
            elif len(shape) == 2 and shape[1] == 4:
                # [batch, 4] - could be boxes for a single detection
                boxes_idx = i
            elif len(shape) == 2 and shape[1] == 1:
                # [batch, 1] - could be classes or scores
                if classes_idx is None:
                    classes_idx = i
                elif scores_idx is None:
                    scores_idx = i
            elif len(shape) == 2:
                # [batch, n] - likely scores
                scores_idx = i
        
        # Extract data based on identified indices
        if boxes_idx is not None and scores_idx is not None:
            boxes = output_data[boxes_idx][0]  # [num_boxes, 4]
            scores = output_data[scores_idx][0]  # [num_boxes]
            
            # Classes tensor might be missing in some models, default to 0 (single class)
            if classes_idx is not None:
                classes = output_data[classes_idx][0]  # [num_boxes]
            else:
                classes = np.zeros_like(scores)
                
            # Filter detections by threshold
            valid_indices = np.where(scores > self.threshold)[0]
            valid_boxes = boxes[valid_indices]
            valid_classes = classes[valid_indices].astype(np.int32)
            valid_scores = scores[valid_indices]
            
            # Convert classes to labels
            valid_labels = []
            for class_id in valid_classes:
                if class_id < len(self.labels):
                    valid_labels.append(self.labels[class_id])
                else:
                    valid_labels.append(f"class_{class_id}")
            
            # Check for weed detections
            weed_detections = []
            for i, label in enumerate(valid_labels):
                if label.lower() == "weed":
                    weed_detections.append({
                        "score": float(valid_scores[i]),
                        "box": valid_boxes[i].tolist()
                    })
            
            return {
                "type": "detection",
                "boxes": valid_boxes.tolist() if isinstance(valid_boxes, np.ndarray) else valid_boxes,
                "classes": valid_classes.tolist() if isinstance(valid_classes, np.ndarray) else valid_classes,
                "labels": valid_labels,
                "scores": valid_scores.tolist() if isinstance(valid_scores, np.ndarray) else valid_scores,
                "weed_detected": len(weed_detections) > 0,
                "weed_detections": weed_detections
            }
        else:
            # Fallback for unknown output format
            self.logger.warning("Unknown detection output format")
            return {
                "type": "unknown",
                "weed_detected": False,
                "error": "Failed to parse model output"
            }
    
    def _run_inference(self, image):
        """Run model inference on a single image.
        
        Args:
            image: RGB numpy array
            
        Returns:
            dict: Detection or classification results
        """
        start_time = time.time()
        
        # Preprocess image
        preprocessed = self._preprocess_image(image)
        preprocess_time = time.time() - start_time
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            preprocessed
        )
        
        # Run inference
        inference_start = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - inference_start
        
        # Get output tensors
        output_data = []
        for output in self.output_details:
            tensor = self.interpreter.get_tensor(output['index'])
            output_data.append(tensor)
        
        # Parse results
        results = self._parse_output(output_data)
        
        # Add timing information
        results["timing"] = {
            "preprocess_ms": preprocess_time * 1000,
            "inference_ms": inference_time * 1000,
            "total_ms": (time.time() - start_time) * 1000
        }
        
        return results
    
    def _continuous_inference_loop(self, image_provider):
        """Continuous inference loop for background thread.
        
        Args:
            image_provider: Function that returns current image when called
        """
        while self.running:
            try:
                # Get current image
                image = image_provider()
                if image is None:
                    time.sleep(0.01)
                    continue
                
                # Run inference
                result = self._run_inference(image)
                
                # Update last result
                with self.result_lock:
                    self.last_result = result
                
                # Short sleep to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in inference loop: {e}")
                time.sleep(0.1)  # Avoid tight error loop
    
    def start_continuous_inference(self, image_provider):
        """Start continuous inference in background thread.
        
        Args:
            image_provider: Function that returns current image when called
            
        Returns:
            bool: Success status
        """
        if not self.enable_threading:
            self.logger.warning("Threading not enabled for inference module")
            return False
            
        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.logger.warning("Inference thread already running")
            return False
            
        self.running = True
        self.inference_thread = threading.Thread(
            target=self._continuous_inference_loop,
            args=(image_provider,)
        )
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        self.logger.info("Started continuous inference thread")
        return True
    
    def stop_continuous_inference(self):
        """Stop continuous inference thread."""
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
            self.inference_thread = None
            
        self.logger.info("Stopped continuous inference thread")
    
    def get_last_result(self):
        """Get most recent inference result from continuous mode.
        
        Returns:
            dict: Last inference result or None
        """
        with self.result_lock:
            return self.last_result
    
    def detect(self, image):
        """Run inference on an image.
        
        In continuous mode, this returns the latest result.
        In direct mode, this runs inference immediately.
        
        Args:
            image: RGB numpy array
            
        Returns:
            dict: Detection results
        """
        if self.continuous_mode and self.inference_thread and self.inference_thread.is_alive():
            # Get latest result from continuous thread
            return self.get_last_result()
        else:
            # Run inference directly
            return self._run_inference(image)
    
    def get_model_info(self):
        """Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            "model_path": self.model_path,
            "input_size": self.input_size,
            "is_quantized": self.is_quantized,
            "threshold": self.threshold,
            "num_labels": len(self.labels),
            "labels": self.labels[:10] + ['...'] if len(self.labels) > 10 else self.labels
        }