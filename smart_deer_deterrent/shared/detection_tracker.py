from collections import defaultdict
import time

class DetectionTracker:
    """Tracks detections across frames to provide persistence and smoothing."""
    
    def __init__(self, persistence_frames=10):
        self.persistence_frames = persistence_frames
        self.detection_history = []
        self.frame_count = 0
        
    def update(self, detections):
        """Update tracker with new detections for current frame."""
        self.frame_count += 1
        
        # Filter out invalid detections and add frame number
        valid_detections = []
        for det in detections:
            if 'box' in det and 'label' in det:
                det['last_seen_frame'] = self.frame_count
                valid_detections.append(det)
            
        # Merge with existing detections
        merged_detections = self._merge_detections(valid_detections)
        
        # Filter out old detections
        self.detection_history = [
            det for det in merged_detections 
            if self.frame_count - det['last_seen_frame'] <= self.persistence_frames
        ]
        
        return self.detection_history
    
    def _merge_detections(self, new_detections):
        """Merge new detections with existing ones, updating positions."""
        from shared.math_utils import calculate_iou
        
        merged = []
        used_indices = set()
        
        # Match new detections with existing ones
        for new_det in new_detections:
            best_match = None
            best_iou = 0
            best_idx = -1
            
            for idx, hist_det in enumerate(self.detection_history):
                if idx in used_indices:
                    continue
                    
                # Skip if either detection lacks required fields
                if 'box' not in new_det or 'box' not in hist_det:
                    continue
                if 'label' not in new_det or 'label' not in hist_det:
                    continue
                    
                # Check if same label and similar position
                if new_det['label'] == hist_det['label']:
                    iou = calculate_iou(new_det['box'], hist_det['box'])
                    if iou > best_iou and iou > 0.3:  # Minimum IoU for match
                        best_match = hist_det
                        best_iou = iou
                        best_idx = idx
            
            if best_match:
                # Update existing detection
                used_indices.add(best_idx)
                updated_det = new_det.copy()
                updated_det['persistence_count'] = best_match.get('persistence_count', 0) + 1
                merged.append(updated_det)
            else:
                # New detection
                new_det['persistence_count'] = 1
                merged.append(new_det)
        
        # Add unmatched historical detections (they persist)
        for idx, hist_det in enumerate(self.detection_history):
            if idx not in used_indices:
                # Decay confidence for persisted detections
                persisted_det = hist_det.copy()
                frames_since_seen = self.frame_count - hist_det['last_seen_frame']
                confidence_decay = 1.0 - (frames_since_seen / self.persistence_frames) * 0.5
                persisted_det['conf'] = hist_det.get('conf', 0.5) * confidence_decay
                persisted_det['persisted'] = True
                # Ensure all required fields are present
                if 'box' not in persisted_det:
                    continue  # Skip if no box info
                if 'label' not in persisted_det:
                    persisted_det['label'] = 'unknown'
                merged.append(persisted_det)
        
        return merged
    
    def reset(self):
        """Reset the tracker."""
        self.detection_history = []
        self.frame_count = 0