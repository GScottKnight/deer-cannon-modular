import sys
import os
import json
from typing import Tuple, Optional, Dict
from dataclasses import asdict

# Adjust the path to include the parent directory (smart_deer_deterrent)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from servo_controller.main import ServoController
from turret_control.coordinate_transform import (
    CoordinateTransformer, CameraConfig, TurretConfig
)
from turret_control.smoothing_filter import TurretSmoothingFilter, AdaptiveSmoothingFilter


class TurretController:
    def __init__(self, camera_config: Optional[CameraConfig] = None,
                 turret_config: Optional[TurretConfig] = None,
                 config_file: Optional[str] = None,
                 use_smoothing: bool = True,
                 use_adaptive_smoothing: bool = False):
        """
        Initialize turret controller with camera awareness.
        
        Args:
            camera_config: Camera configuration object
            turret_config: Turret configuration object  
            config_file: Path to JSON config file (overrides other configs)
            use_smoothing: Enable smoothing filter for servo control
            use_adaptive_smoothing: Use adaptive smoothing that adjusts to movement
        """
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self._load_config(config_file)
        else:
            self.camera_config = camera_config or CameraConfig()
            self.turret_config = turret_config or TurretConfig()
        
        # Initialize coordinate transformer
        self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
        
        # Initialize servo controller (placeholder for now)
        self.servo_controller = ServoController()
        
        # Initialize smoothing filter
        self.use_smoothing = use_smoothing
        if use_smoothing:
            if use_adaptive_smoothing:
                self.smoothing_filter = AdaptiveSmoothingFilter(
                    position_alpha=0.1,  # Much more smoothing (was 0.3)
                    angle_alpha=0.05,    # Very heavy angle smoothing (was 0.2)
                    velocity_alpha=0.2,  # More velocity smoothing (was 0.4)
                    max_velocity=45.0    # Reduced max velocity (was 90.0)
                )
            else:
                self.smoothing_filter = TurretSmoothingFilter(
                    position_alpha=0.1,  # Much more smoothing (was 0.3)
                    angle_alpha=0.05,    # Very heavy angle smoothing (was 0.2)
                    velocity_alpha=0.2,  # More velocity smoothing (was 0.4)
                    max_velocity=45.0    # Reduced max velocity (was 90.0)
                )
        else:
            self.smoothing_filter = None
        
        # Tracking state
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.is_tracking = False
        self.last_target = None

    def _load_config(self, config_file: str):
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Create config objects from JSON
        self.camera_config = CameraConfig(**config.get('camera', {}))
        self.turret_config = TurretConfig(**config.get('turret', {}))
    
    def save_config(self, config_file: str):
        """Save current configuration to JSON file."""
        config = {
            'camera': asdict(self.camera_config),
            'turret': asdict(self.turret_config)
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def update_camera_pan(self, pan_angle: float):
        """Update the camera's current pan position."""
        self.camera_config.pan_position = pan_angle
        self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
    
    def aim_at_target(self, target_box, apply_parallax: bool = True):
        """
        Calculate and execute turret aim at target.
        
        Args:
            target_box: [x1, y1, x2, y2] bounding box
            apply_parallax: Whether to apply parallax correction
        """
        # Apply bbox-based smoothing if enabled
        if self.use_smoothing and self.smoothing_filter:
            # Use corner tracking to find optimal aim point
            target_center_x, target_center_y = self.smoothing_filter.smooth_bbox(target_box)
        else:
            # Calculate the center of the bounding box
            target_center_x = (target_box[0] + target_box[2]) / 2
            target_center_y = (target_box[1] + target_box[3]) / 2
        
        # Transform coordinates to turret angles
        pan_angle, tilt_angle, is_valid = self.transformer.transform(
            target_center_x, target_center_y, apply_parallax
        )
        
        # Apply angle smoothing if enabled
        if self.use_smoothing and self.smoothing_filter:
            pan_angle, tilt_angle = self.smoothing_filter.smooth_angles(
                pan_angle, tilt_angle
            )
        
        if not is_valid:
            print(f"WARNING: Target outside turret limits!")
        
        # Get transformation details for logging
        details = self.transformer.get_transformation_details(
            target_center_x, target_center_y
        )
        
        print(f"Targeting: Camera pan={self.camera_config.pan_position:.1f}°, "
              f"Target at ({target_center_x:.0f}, {target_center_y:.0f})")
        print(f"  Camera-relative: pan={details['camera_relative']['pan']:.2f}°, "
              f"tilt={details['camera_relative']['tilt']:.2f}°")
        print(f"  Turret command: pan={pan_angle:.2f}°, tilt={tilt_angle:.2f}°")
        
        # Store current position
        self.current_pan = pan_angle
        self.current_tilt = tilt_angle
        self.last_target = target_box
        
        # Send to servo controller
        self.servo_controller.aim_at(pan_angle, tilt_angle)
    
    def aim_at_pixel(self, pixel_x: float, pixel_y: float, apply_parallax: bool = True):
        """
        Aim at specific pixel coordinates.
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            apply_parallax: Whether to apply parallax correction
        """
        # Apply position smoothing if enabled
        if self.use_smoothing and self.smoothing_filter:
            pixel_x, pixel_y = self.smoothing_filter.smooth_position(pixel_x, pixel_y)
        
        # Transform coordinates to turret angles
        pan_angle, tilt_angle, is_valid = self.transformer.transform(
            pixel_x, pixel_y, apply_parallax
        )
        
        # Apply angle smoothing if enabled
        if self.use_smoothing and self.smoothing_filter:
            pan_angle, tilt_angle = self.smoothing_filter.smooth_angles(
                pan_angle, tilt_angle
            )
        
        if not is_valid:
            print(f"WARNING: Target outside turret limits!")
        
        print(f"Aiming at pixel ({pixel_x:.0f}, {pixel_y:.0f}): "
              f"pan={pan_angle:.2f}°, tilt={tilt_angle:.2f}°")
        
        # Store current position
        self.current_pan = pan_angle
        self.current_tilt = tilt_angle
        
        # Send to servo controller
        self.servo_controller.aim_at(pan_angle, tilt_angle)
    
    def get_turret_info(self) -> Dict:
        """Get current turret information."""
        info = {
            'position': {
                'pan': self.current_pan,
                'tilt': self.current_tilt
            },
            'camera': {
                'pan': self.camera_config.pan_position,
                'fov_h': self.camera_config.horizontal_fov,
                'fov_v': self.camera_config.vertical_fov
            },
            'limits': {
                'pan_min': self.turret_config.pan_min,
                'pan_max': self.turret_config.pan_max,
                'tilt_min': self.turret_config.tilt_min,
                'tilt_max': self.turret_config.tilt_max
            },
            'is_tracking': self.is_tracking,
            'last_target': self.last_target
        }
        
        # Add corner tracking info if available
        if self.smoothing_filter:
            corner_info = self.smoothing_filter.get_corner_info()
            if corner_info:
                info['corner_tracking'] = corner_info
        
        return info
    
    def calibrate_offset(self, measured_pan: float, measured_tilt: float,
                        pixel_x: float, pixel_y: float):
        """
        Calibrate turret offset based on measured vs calculated angles.
        
        Args:
            measured_pan: Actual measured pan angle
            measured_tilt: Actual measured tilt angle
            pixel_x: Pixel X that was targeted
            pixel_y: Pixel Y that was targeted
        """
        # Calculate what we thought the angles should be
        calculated_pan, calculated_tilt, _ = self.transformer.transform(
            pixel_x, pixel_y, apply_parallax=True
        )
        
        # Calculate offset
        pan_offset = measured_pan - calculated_pan
        tilt_offset = measured_tilt - calculated_tilt
        
        print(f"Calibration offset: pan={pan_offset:.2f}°, tilt={tilt_offset:.2f}°")
        
        # This could be used to adjust the turret configuration
        # For now, just return the offset
        return pan_offset, tilt_offset
    
    def reset_smoothing(self):
        """Reset smoothing filter state."""
        if self.smoothing_filter:
            self.smoothing_filter.reset()
    
    def set_smoothing_params(self, position_alpha: float = None, 
                           angle_alpha: float = None,
                           max_velocity: float = None):
        """
        Update smoothing parameters.
        
        Args:
            position_alpha: Smoothing factor for position (0-1)
            angle_alpha: Smoothing factor for angles (0-1)
            max_velocity: Maximum angular velocity (degrees/second)
        """
        if not self.smoothing_filter:
            return
        
        if position_alpha is not None:
            self.smoothing_filter.position_x_filter.alpha = position_alpha
            self.smoothing_filter.position_y_filter.alpha = position_alpha
        
        if angle_alpha is not None:
            self.smoothing_filter.pan_filter.alpha = angle_alpha
            self.smoothing_filter.tilt_filter.alpha = angle_alpha
        
        if max_velocity is not None:
            self.smoothing_filter.max_velocity = max_velocity
