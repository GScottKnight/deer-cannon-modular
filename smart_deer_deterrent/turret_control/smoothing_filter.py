"""
Smoothing filter for turret servo control.

This module provides filtering to smooth noisy detection data and prevent
servo jitter and oscillation.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import deque
import time
from enum import Enum


class Corner(Enum):
    """Enum for bounding box corners."""
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


class CornerTracker:
    """Tracks stability of bounding box corners."""
    
    def __init__(self, history_size: int = 10, stability_threshold: float = 10.0):
        """
        Initialize corner tracker.
        
        Args:
            history_size: Number of frames to track
            stability_threshold: Max movement in pixels to consider stable
        """
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        
        # History for each corner
        self.corner_history = {
            Corner.TOP_LEFT: deque(maxlen=history_size),
            Corner.TOP_RIGHT: deque(maxlen=history_size),
            Corner.BOTTOM_LEFT: deque(maxlen=history_size),
            Corner.BOTTOM_RIGHT: deque(maxlen=history_size)
        }
        
        # Stability scores (lower = more stable)
        self.stability_scores = {
            Corner.TOP_LEFT: float('inf'),
            Corner.TOP_RIGHT: float('inf'),
            Corner.BOTTOM_LEFT: float('inf'),
            Corner.BOTTOM_RIGHT: float('inf')
        }
        
        # Smoothed corner positions
        self.smoothed_corners = {
            Corner.TOP_LEFT: None,
            Corner.TOP_RIGHT: None,
            Corner.BOTTOM_LEFT: None,
            Corner.BOTTOM_RIGHT: None
        }
    
    def update(self, bbox: List[float]) -> Dict[Corner, Tuple[float, float]]:
        """
        Update corner tracking with new bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            Dictionary of corner positions
        """
        x1, y1, x2, y2 = bbox
        
        # Extract corners
        corners = {
            Corner.TOP_LEFT: (x1, y1),
            Corner.TOP_RIGHT: (x2, y1),
            Corner.BOTTOM_LEFT: (x1, y2),
            Corner.BOTTOM_RIGHT: (x2, y2)
        }
        
        # Update history and calculate stability
        for corner, pos in corners.items():
            self.corner_history[corner].append(pos)
            
            # Calculate stability score if we have enough history
            if len(self.corner_history[corner]) >= 3:
                self._calculate_stability(corner)
            
            # Update smoothed position
            if self.smoothed_corners[corner] is None:
                self.smoothed_corners[corner] = pos
            else:
                # Simple exponential smoothing
                alpha = 0.3
                self.smoothed_corners[corner] = (
                    alpha * pos[0] + (1 - alpha) * self.smoothed_corners[corner][0],
                    alpha * pos[1] + (1 - alpha) * self.smoothed_corners[corner][1]
                )
        
        return corners
    
    def _calculate_stability(self, corner: Corner):
        """Calculate stability score for a corner."""
        history = list(self.corner_history[corner])
        
        # Calculate movement variance
        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        
        # Use standard deviation as stability metric
        std_x = np.std(xs) if len(xs) > 1 else 0
        std_y = np.std(ys) if len(ys) > 1 else 0
        
        # Combined stability score
        self.stability_scores[corner] = np.sqrt(std_x**2 + std_y**2)
    
    def get_stable_corners(self) -> List[Corner]:
        """Get list of corners that are considered stable."""
        stable = []
        for corner, score in self.stability_scores.items():
            if score < self.stability_threshold:
                stable.append(corner)
        return stable
    
    def get_most_stable_corner(self) -> Optional[Corner]:
        """Get the most stable corner."""
        if all(s == float('inf') for s in self.stability_scores.values()):
            return None
        
        return min(self.stability_scores.items(), key=lambda x: x[1])[0]
    
    def get_optimal_aim_point(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calculate optimal aim point based on bottom edge stability.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (x, y) optimal aim point
        """
        x1, y1, x2, y2 = bbox
        
        # Always use the bottom edge as the stable reference
        # This works because animals' feet stay planted when feeding
        
        # X coordinate: center of bottom edge
        x = (x1 + x2) / 2
        
        # Y coordinate: aim at a fixed percentage up from the bottom
        # This targets the body mass, not the moving head
        box_height = y2 - y1
        
        # Aim at 30% up from the bottom (typically hits center of mass)
        # This stays stable even when the animal's head moves up/down
        y = y2 - (box_height * 0.3)
        
        return (x, y)


class ExponentialMovingAverage:
    """Exponential Moving Average filter for single values."""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize EMA filter.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother
        """
        self.alpha = alpha
        self.value: Optional[float] = None
        self.last_update = time.time()
    
    def update(self, new_value: float) -> float:
        """
        Update filter with new value.
        
        Args:
            new_value: New measurement
            
        Returns:
            Smoothed value
        """
        current_time = time.time()
        
        if self.value is None:
            self.value = new_value
        else:
            # Adjust alpha based on time since last update
            # This helps handle variable frame rates
            dt = current_time - self.last_update
            adjusted_alpha = 1 - (1 - self.alpha) ** dt
            
            self.value = adjusted_alpha * new_value + (1 - adjusted_alpha) * self.value
        
        self.last_update = current_time
        return self.value
    
    def reset(self):
        """Reset filter state."""
        self.value = None
        self.last_update = time.time()


class TurretSmoothingFilter:
    """Complete smoothing solution for turret control."""
    
    def __init__(self, position_alpha: float = 0.3, angle_alpha: float = 0.2,
                 velocity_alpha: float = 0.4, max_velocity: float = 180.0,
                 use_corner_tracking: bool = True):
        """
        Initialize turret smoothing filter.
        
        Args:
            position_alpha: Smoothing for detection position (pixels)
            angle_alpha: Smoothing for calculated angles (degrees)
            velocity_alpha: Smoothing for velocity estimation
            max_velocity: Maximum angular velocity (degrees/second)
            use_corner_tracking: Enable corner-based stability detection
        """
        # Position smoothing (for detection center)
        self.position_x_filter = ExponentialMovingAverage(position_alpha)
        self.position_y_filter = ExponentialMovingAverage(position_alpha)
        
        # Angle smoothing
        self.pan_filter = ExponentialMovingAverage(angle_alpha)
        self.tilt_filter = ExponentialMovingAverage(angle_alpha)
        
        # Velocity estimation for predictive smoothing
        self.pan_velocity_filter = ExponentialMovingAverage(velocity_alpha)
        self.tilt_velocity_filter = ExponentialMovingAverage(velocity_alpha)
        
        # State tracking
        self.last_pan: Optional[float] = None
        self.last_tilt: Optional[float] = None
        self.last_update = time.time()
        self.max_velocity = max_velocity
        
        # Deadband to ignore small movements
        self.position_deadband = 15.0  # pixels (increased from 5.0 for more stability)
        self.angle_deadband = 1.5      # degrees (increased from 0.5 for less jitter)
        
        # Corner tracking
        self.use_corner_tracking = use_corner_tracking
        if use_corner_tracking:
            self.corner_tracker = CornerTracker(history_size=10, stability_threshold=10.0)
        else:
            self.corner_tracker = None
        
        # Store last bbox for corner tracking
        self.last_bbox = None
        
        # Bottom edge tracking - use heavy smoothing for Y coordinate
        self.bottom_edge_filter = ExponentialMovingAverage(0.05)  # Very heavy smoothing
    
    def smooth_bbox(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Smooth position using bottom-edge tracking for stationary feeding animals.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            (x, y) smoothed aim point
        """
        x1, y1, x2, y2 = bbox
        
        # Update corner tracking data (for debugging/visualization)
        if self.corner_tracker:
            self.corner_tracker.update(bbox)
        self.last_bbox = bbox
        
        # Bottom-edge based tracking
        # X: Center of bottom edge with standard smoothing
        bottom_center_x = (x1 + x2) / 2
        smoothed_x = self.position_x_filter.update(bottom_center_x)
        
        # Y: Track bottom edge with heavy smoothing
        bottom_y = y2
        smoothed_bottom_y = self.bottom_edge_filter.update(bottom_y)
        
        # Calculate height from smoothed bottom edge
        current_height = y2 - y1
        
        # Aim point: 30% up from the smoothed bottom edge
        # This keeps the aim point stable even when the head moves
        aim_y = smoothed_bottom_y - (current_height * 0.3)
        
        # Apply final position smoothing with deadband
        return self.smooth_position(smoothed_x, aim_y)
    
    def smooth_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Smooth detection position.
        
        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            
        Returns:
            (smoothed_x, smoothed_y)
        """
        # Apply deadband with larger Y deadband for feeding animals
        if self.position_x_filter.value is not None:
            if abs(x - self.position_x_filter.value) < self.position_deadband:
                x = self.position_x_filter.value
        
        # Use larger deadband for Y to reduce vertical jitter
        y_deadband = self.position_deadband * 2.0  # Double deadband for Y
        if self.position_y_filter.value is not None:
            if abs(y - self.position_y_filter.value) < y_deadband:
                y = self.position_y_filter.value
        
        smoothed_x = self.position_x_filter.update(x)
        smoothed_y = self.position_y_filter.update(y)
        
        return smoothed_x, smoothed_y
    
    def smooth_angles(self, pan: float, tilt: float) -> Tuple[float, float]:
        """
        Smooth turret angles with velocity limiting.
        
        Args:
            pan: Raw pan angle in degrees
            tilt: Raw tilt angle in degrees
            
        Returns:
            (smoothed_pan, smoothed_tilt)
        """
        current_time = time.time()
        dt = current_time - self.last_update
        
        # Calculate velocities if we have previous values
        if self.last_pan is not None and dt > 0:
            pan_velocity = (pan - self.last_pan) / dt
            tilt_velocity = (tilt - self.last_tilt) / dt
            
            # Limit velocities
            pan_velocity = np.clip(pan_velocity, -self.max_velocity, self.max_velocity)
            tilt_velocity = np.clip(tilt_velocity, -self.max_velocity, self.max_velocity)
            
            # Smooth velocities
            smoothed_pan_vel = self.pan_velocity_filter.update(pan_velocity)
            smoothed_tilt_vel = self.tilt_velocity_filter.update(tilt_velocity)
            
            # Apply velocity-based prediction
            predicted_pan = self.last_pan + smoothed_pan_vel * dt
            predicted_tilt = self.last_tilt + smoothed_tilt_vel * dt
            
            # Blend prediction with measurement
            pan = 0.7 * pan + 0.3 * predicted_pan
            tilt = 0.7 * tilt + 0.3 * predicted_tilt
        
        # Apply deadband
        if self.pan_filter.value is not None:
            if abs(pan - self.pan_filter.value) < self.angle_deadband:
                pan = self.pan_filter.value
        
        if self.tilt_filter.value is not None:
            if abs(tilt - self.tilt_filter.value) < self.angle_deadband:
                tilt = self.tilt_filter.value
        
        # Smooth angles
        smoothed_pan = self.pan_filter.update(pan)
        smoothed_tilt = self.tilt_filter.update(tilt)
        
        # Update state
        self.last_pan = smoothed_pan
        self.last_tilt = smoothed_tilt
        self.last_update = current_time
        
        return smoothed_pan, smoothed_tilt
    
    def reset(self):
        """Reset all filters."""
        self.position_x_filter.reset()
        self.position_y_filter.reset()
        self.pan_filter.reset()
        self.tilt_filter.reset()
        self.pan_velocity_filter.reset()
        self.tilt_velocity_filter.reset()
        self.last_pan = None
        self.last_tilt = None
        self.last_update = time.time()
        
        # Reset corner tracker
        if self.corner_tracker:
            self.corner_tracker = CornerTracker(
                history_size=self.corner_tracker.history_size,
                stability_threshold=self.corner_tracker.stability_threshold
            )
        self.last_bbox = None
        
        # Reset bottom edge filter
        self.bottom_edge_filter.reset()
    
    def get_corner_info(self) -> Optional[Dict]:
        """Get corner tracking information for debugging."""
        if not self.corner_tracker or not self.last_bbox:
            return None
        
        stable_corners = self.corner_tracker.get_stable_corners()
        most_stable = self.corner_tracker.get_most_stable_corner()
        
        return {
            'stable_corners': [c.name for c in stable_corners],
            'most_stable': most_stable.name if most_stable else None,
            'stability_scores': {
                c.name: score for c, score in self.corner_tracker.stability_scores.items()
            },
            'using_corner_tracking': self.use_corner_tracking
        }


class AdaptiveSmoothingFilter(TurretSmoothingFilter):
    """Adaptive smoothing that adjusts based on movement characteristics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.movement_detector = MovementDetector()
    
    def smooth_angles(self, pan: float, tilt: float) -> Tuple[float, float]:
        """
        Smooth angles with adaptive filtering based on movement.
        
        Args:
            pan: Raw pan angle in degrees
            tilt: Raw tilt angle in degrees
            
        Returns:
            (smoothed_pan, smoothed_tilt)
        """
        # Detect movement characteristics
        is_fast = self.movement_detector.is_fast_movement(pan, tilt)
        is_stationary = self.movement_detector.is_stationary(pan, tilt)
        
        # Adjust smoothing parameters
        if is_fast:
            # Less smoothing for fast movements to maintain responsiveness
            self.pan_filter.alpha = 0.3  # Still some smoothing even for fast movement
            self.tilt_filter.alpha = 0.3
        elif is_stationary:
            # Much more smoothing when nearly stationary to eliminate jitter
            self.pan_filter.alpha = 0.02  # Very heavy smoothing for stationary targets
            self.tilt_filter.alpha = 0.02
            # Also increase deadbands for stationary targets
            self.position_deadband = 25.0  # Ignore movements up to 25 pixels
            self.angle_deadband = 2.5      # Ignore angle changes up to 2.5 degrees
        else:
            # Normal smoothing for moderate movements
            self.pan_filter.alpha = 0.1   # More smoothing than before
            self.tilt_filter.alpha = 0.1
            # Reset deadbands to normal
            self.position_deadband = 15.0
            self.angle_deadband = 1.5
        
        return super().smooth_angles(pan, tilt)


class MovementDetector:
    """Detect movement characteristics for adaptive filtering."""
    
    def __init__(self, window_size: int = 5):
        self.pan_history = deque(maxlen=window_size)
        self.tilt_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
    
    def update(self, pan: float, tilt: float):
        """Update movement history."""
        self.pan_history.append(pan)
        self.tilt_history.append(tilt)
        self.time_history.append(time.time())
    
    def is_fast_movement(self, pan: float, tilt: float, 
                        threshold: float = 20.0) -> bool:  # Reduced from 30.0
        """Check if target is moving fast."""
        self.update(pan, tilt)
        
        if len(self.pan_history) < 2:
            return False
        
        # Calculate average velocity
        dt = self.time_history[-1] - self.time_history[0]
        if dt <= 0:
            return False
        
        pan_vel = abs(self.pan_history[-1] - self.pan_history[0]) / dt
        tilt_vel = abs(self.tilt_history[-1] - self.tilt_history[0]) / dt
        
        return pan_vel > threshold or tilt_vel > threshold
    
    def is_stationary(self, pan: float, tilt: float,
                     threshold: float = 5.0) -> bool:  # Increased from 2.0 to be more inclusive
        """Check if target is nearly stationary."""
        self.update(pan, tilt)
        
        if len(self.pan_history) < self.pan_history.maxlen:
            return True  # Assume stationary until we have enough history
        
        # Check variance in recent positions
        pan_var = np.var(list(self.pan_history))
        tilt_var = np.var(list(self.tilt_history))
        
        # Also check range of motion
        pan_range = max(self.pan_history) - min(self.pan_history)
        tilt_range = max(self.tilt_history) - min(self.tilt_history)
        
        # Consider stationary if variance is low OR range is small
        return (pan_var < threshold and tilt_var < threshold) or (pan_range < 3.0 and tilt_range < 3.0)