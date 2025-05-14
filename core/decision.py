#core/decision.py
import logging
import time
class DecisionModule:
    """Core module for making decisions based on AI inference results.
    This class interprets the output from the AI inference module and
determines what actions the robot should take.
"""

def __init__(self, config):
    """Initialize decision module.
    
    Args:
        config: Dict with decision module settings
            - threshold: Confidence threshold for actions
            - cooldown_period: Minimum time between actions (seconds)
            - max_spray_count: Maximum sprays before requiring movement
            - position_weighting: Whether to weight central detections higher
            - delay_after_detection: Delay after detection before action (seconds)
    """
    self.logger = logging.getLogger("Decision")
    self.config = config
    
    # Decision parameters
    self.threshold = config.get('threshold', 0.5)
    self.cooldown_period = config.get('cooldown_period', 2.0)
    self.max_spray_count = config.get('max_spray_count', 5)
    self.position_weighting = config.get('position_weighting', True)
    self.delay_after_detection = config.get('delay_after_detection', 0.2)
    
    # State tracking
    self.last_action_time = 0
    self.spray_count = 0
    self.last_decision = None
    
def interpret_detection(self, detection_result):
    """Interpret AI detection results and decide action.
    
    Args:
        detection_result: Dict from AIInferenceModule.detect()
        
    Returns:
        dict: Decision with actions to take
    """
    current_time = time.time()
    decision = {
        'weed_detected': False,
        'action': 'CONTINUE',  # CONTINUE, STOP, SPRAY
        'confidence': 0.0,
        'timestamp': current_time,
        'position': None,
        'class_name': None
    }
    
    # Check cooldown period
    if current_time - self.last_action_time < self.cooldown_period:
        self.logger.debug("In cooldown period, continuing")
        return decision
        
    # Handle classification model output
    if detection_result['type'] == 'classification':
        self._handle_classification(detection_result, decision)
    
    # Handle object detection model output
    elif detection_result['type'] == 'detection':
        self._handle_detection(detection_result, decision)
    
    # Apply additional decision logic
    self._apply_decision_policy(decision)
    
    # Update state
    if decision['action'] != 'CONTINUE':
        self.last_action_time = current_time
        if decision['action'] == 'SPRAY':
            self.spray_count += 1
    
    self.last_decision = decision
    return decision
    
def _handle_classification(self, detection_result, decision):
    """Process classification model output.
    
    Args:
        detection_result: Classification result dict
        decision: Decision dict to update
    """
    label = detection_result['label']
    score = detection_result['score']
    
    # Update decision with detection info
    decision['class_name'] = label
    decision['confidence'] = score
    
    # Check if prediction is "weed" with sufficient confidence
    if 'weed' in label.lower() and score > self.threshold:
        decision['weed_detected'] = True
        decision['action'] = 'SPRAY'
        self.logger.info(f"Weed classification: {label} ({score:.2f})")

def _handle_detection(self, detection_result, decision):
    """Process object detection model output.
    
    Args:
        detection_result: Detection result dict
        decision: Decision dict to update
    """
    # Check if any detected object is a weed
    weed_detections = []
    
    for i, label in enumerate(detection_result['labels']):
        if 'weed' in label.lower():
            # Get the detection box [y1, x1, y2, x2] or [x1, y1, x2, y2] depending on model
            box = detection_result['boxes'][i]
            score = detection_result['scores'][i]
            
            # Apply position weighting if enabled (favor center detections)
            if self.position_weighting:
                # Calculate how central the detection is (0-1 where 1 is center)
                # Assuming box format [y1, x1, y2, x2]
                box_center_y = (box[0] + box[2]) / 2
                box_center_x = (box[1] + box[3]) / 2
                
                # Distance from center (0 = center, 1 = corner)
                center_dist = ((box_center_x - 0.5)**2 + (box_center_y - 0.5)**2)**0.5
                center_weight = max(0, 1 - center_dist)
                
                # Apply modest boost to centrally located weeds (max 20% boost)
                weighted_score = score * (1 + (0.2 * center_weight))
                
                self.logger.debug(f"Weighting applied: {score:.2f} -> {weighted_score:.2f}")
                score = min(weighted_score, 1.0)  # Cap at 1.0
            
            weed_detections.append({
                'label': label,
                'score': score,
                'box': box
            })
    
    # If any weeds detected with sufficient confidence
    if weed_detections:
        # Get the detection with highest confidence
        best_detection = max(weed_detections, key=lambda x: x['score'])
        
        if best_detection['score'] > self.threshold:
            decision['weed_detected'] = True
            decision['action'] = 'SPRAY'
            decision['confidence'] = best_detection['score']
            decision['position'] = best_detection['box']
            decision['class_name'] = best_detection['label']
            
            self.logger.info(f"Weed detection: {best_detection['label']} " 
                            f"({best_detection['score']:.2f}) at {best_detection['box']}")

def _apply_decision_policy(self, decision):
    """Apply additional policy rules to the decision.
    
    Args:
        decision: Decision dict to modify
    """
    # If we've sprayed too many times without moving, force movement
    if decision['action'] == 'SPRAY' and self.spray_count >= self.max_spray_count:
        self.logger.info(f"Max spray count ({self.max_spray_count}) reached, forcing movement")
        decision['action'] = 'CONTINUE'
        decision['weed_detected'] = False
        self.spray_count = 0
    
def reset_state(self):
    """Reset the internal state of the decision module."""
    self.last_action_time = 0
    self.spray_count = 0
    self.last_decision = None
    self.logger.info("Decision state reset")