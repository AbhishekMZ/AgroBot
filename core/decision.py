# core/decision.py
class DecisionModule:
    def __init__(self, config):
        """Initialize the decision-making logic.
        
        Args:
            config: Dict with decision thresholds and settings
        """
        self.config = config
        
    def interpret_detection(self, detection_result):
        """Interpret AI detection results and decide action.
        
        Args:
            detection_result: Dict from AIInferenceModule.detect()
            
        Returns:
            dict: Decision with actions to take
        """
        decision = {
            'weed_detected': False,
            'action': 'CONTINUE',  # CONTINUE, STOP, SPRAY
            'confidence': 0.0
        }
        
        # Handle classification model output
        if detection_result['type'] == 'classification':
            label = detection_result['label']
            score = detection_result['score']
            
            # Check if prediction is "weed" with sufficient confidence
            if label.lower() == 'weed' and score > self.config['threshold']:
                decision['weed_detected'] = True
                decision['action'] = 'SPRAY'
                decision['confidence'] = score
        
        # Handle object detection model output
        elif detection_result['type'] == 'detection':
            # Check if any detected object is a weed
            weed_detections = []
            
            for i, label in enumerate(detection_result['labels']):
                if label.lower() == 'weed':
                    weed_detections.append({
                        'score': detection_result['scores'][i],
                        'box': detection_result['boxes'][i]
                    })
            
            # If any weeds detected with sufficient confidence
            if weed_detections and max(d['score'] for d in weed_detections) > self.config['threshold']:
                best_detection = max(weed_detections, key=lambda x: x['score'])
                decision['weed_detected'] = True
                decision['action'] = 'SPRAY'
                decision['confidence'] = best_detection['score']
                decision['position'] = best_detection['box']  # For positioning sprayer
        
        return decision