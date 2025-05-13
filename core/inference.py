# core/inference.py
import numpy as np
import tensorflow as tf
import cv2
import time

class AIInferenceModule:
    def __init__(self, config):
        """Initialize TFLite inference module.
        
        Args:
            config: Dict with model settings
        """
        self.model_path = config['path']
        self.input_size = tuple(config['input_size'])
        self.threshold = config['threshold']
        
        # Load labels
        self.labels = []
        with open(config['labels'], 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Log model info
        print(f"Loaded model: {self.model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input type: {self.input_details[0]['dtype']}")
        
    def preprocess_image(self, image):
        """Preprocess image for model input.
        
        Args:
            image: RGB numpy array
            
        Returns:
            preprocessed_image: Resized and normalized numpy array
        """
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Normalize to [0,1] or [-1,1] depending on your model requirements
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def detect(self, image):
        """Run inference on an image.
        
        Args:
            image: RGB numpy array
            
        Returns:
            dict: Detection results with classes, scores, and timing
        """
        start_time = time.time()
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            preprocessed.astype(self.input_details[0]['dtype'])
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get results
        # Note: Adjust based on your model output format (classification vs detection)
        if len(self.output_details) >= 2:  # Likely object detection
            # Object detection (e.g., SSD) typically has boxes and scores
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Filter by threshold
            valid_indices = np.where(scores > self.threshold)[0]
            valid_boxes = boxes[valid_indices]
            valid_classes = classes[valid_indices].astype(np.int32)
            valid_scores = scores[valid_indices]
            
            # Convert classes to labels
            valid_labels = [self.labels[idx] if idx < len(self.labels) else f"Unknown_{idx}" 
                           for idx in valid_classes]
            
            results = {
                'type': 'detection',
                'boxes': valid_boxes,
                'classes': valid_classes,
                'labels': valid_labels,
                'scores': valid_scores,
                'time_ms': (time.time() - start_time) * 1000
            }
        else:  # Likely classification
            # Single classification output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Get top prediction
            top_idx = np.argmax(output)
            top_score = output[top_idx]
            
            results = {
                'type': 'classification',
                'class_id': top_idx,
                'label': self.labels[top_idx] if top_idx < len(self.labels) else f"Unknown_{top_idx}",
                'score': top_score,
                'all_scores': output,
                'time_ms': (time.time() - start_time) * 1000
            }
        
        return results