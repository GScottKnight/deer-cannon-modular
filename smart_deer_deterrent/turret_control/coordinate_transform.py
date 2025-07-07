"""
Coordinate transformation module for camera-to-turret targeting.

This module handles the transformation of detected objects in camera space
to absolute turret angles, accounting for camera pan position and physical offsets.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    horizontal_fov: float = 60.0  # degrees
    vertical_fov: float = 40.0    # degrees
    image_width: int = 1280
    image_height: int = 720
    pan_position: float = 0.0     # Current pan angle in degrees


@dataclass
class TurretConfig:
    """Turret configuration parameters."""
    # Physical offset from camera to turret (meters)
    offset_x: float = 0.0  # Left/right offset
    offset_y: float = 0.5  # Up/down offset (turret below camera)
    offset_z: float = 0.0  # Forward/back offset
    
    # Turret limits (degrees)
    pan_min: float = -90.0
    pan_max: float = 90.0
    tilt_min: float = -30.0
    tilt_max: float = 45.0
    
    # Distance to target (meters) - for parallax correction
    target_distance: float = 10.0


class CoordinateTransformer:
    """Transforms camera coordinates to turret angles."""
    
    def __init__(self, camera_config: CameraConfig, turret_config: TurretConfig):
        self.camera = camera_config
        self.turret = turret_config
        
        # Calculate degrees per pixel
        self.deg_per_pixel_x = self.camera.horizontal_fov / self.camera.image_width
        self.deg_per_pixel_y = self.camera.vertical_fov / self.camera.image_height
    
    def pixel_to_camera_angles(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to camera-relative angles.
        
        Args:
            pixel_x: X coordinate in image (0 to image_width)
            pixel_y: Y coordinate in image (0 to image_height)
            
        Returns:
            (pan_angle, tilt_angle) relative to camera center in degrees
        """
        # Calculate pixel offset from center
        center_x = self.camera.image_width / 2
        center_y = self.camera.image_height / 2
        
        pixel_offset_x = pixel_x - center_x
        pixel_offset_y = pixel_y - center_y
        
        # Convert to angles (positive pan = right, positive tilt = up)
        pan_angle = pixel_offset_x * self.deg_per_pixel_x
        tilt_angle = -pixel_offset_y * self.deg_per_pixel_y  # Invert Y axis
        
        return pan_angle, tilt_angle
    
    def camera_to_world_angles(self, camera_pan: float, camera_tilt: float) -> Tuple[float, float]:
        """
        Convert camera-relative angles to world angles.
        
        Args:
            camera_pan: Pan angle relative to camera (degrees)
            camera_tilt: Tilt angle relative to camera (degrees)
            
        Returns:
            (world_pan, world_tilt) in degrees
        """
        # Add camera's current pan position to get world angle
        world_pan = self.camera.pan_position + camera_pan
        world_tilt = camera_tilt  # Assuming camera doesn't tilt
        
        return world_pan, world_tilt
    
    def apply_parallax_correction(self, world_pan: float, world_tilt: float) -> Tuple[float, float]:
        """
        Apply parallax correction for camera-turret offset.
        
        This accounts for the physical separation between camera and turret.
        
        Args:
            world_pan: World pan angle (degrees)
            world_tilt: World tilt angle (degrees)
            
        Returns:
            (corrected_pan, corrected_tilt) in degrees
        """
        # Convert to radians for calculations
        pan_rad = math.radians(world_pan)
        tilt_rad = math.radians(world_tilt)
        
        # Calculate target position in 3D space (assuming target at configured distance)
        target_x = self.turret.target_distance * math.sin(pan_rad) * math.cos(tilt_rad)
        target_y = self.turret.target_distance * math.sin(tilt_rad)
        target_z = self.turret.target_distance * math.cos(pan_rad) * math.cos(tilt_rad)
        
        # Adjust for turret offset
        adjusted_x = target_x - self.turret.offset_x
        adjusted_y = target_y - self.turret.offset_y
        adjusted_z = target_z - self.turret.offset_z
        
        # Calculate new angles from turret perspective
        distance_xz = math.sqrt(adjusted_x**2 + adjusted_z**2)
        
        if distance_xz > 0:
            turret_pan = math.degrees(math.atan2(adjusted_x, adjusted_z))
        else:
            turret_pan = 0.0
            
        turret_tilt = math.degrees(math.atan2(adjusted_y, distance_xz))
        
        return turret_pan, turret_tilt
    
    def constrain_angles(self, pan: float, tilt: float) -> Tuple[float, float]:
        """
        Constrain turret angles to physical limits.
        
        Args:
            pan: Desired pan angle (degrees)
            tilt: Desired tilt angle (degrees)
            
        Returns:
            (constrained_pan, constrained_tilt) in degrees
        """
        constrained_pan = max(self.turret.pan_min, min(self.turret.pan_max, pan))
        constrained_tilt = max(self.turret.tilt_min, min(self.turret.tilt_max, tilt))
        
        return constrained_pan, constrained_tilt
    
    def transform(self, pixel_x: float, pixel_y: float, 
                  apply_parallax: bool = True) -> Tuple[float, float, bool]:
        """
        Complete transformation from pixel coordinates to turret angles.
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            apply_parallax: Whether to apply parallax correction
            
        Returns:
            (turret_pan, turret_tilt, is_valid) where is_valid indicates
            if the target is within turret limits
        """
        # Step 1: Pixel to camera angles
        camera_pan, camera_tilt = self.pixel_to_camera_angles(pixel_x, pixel_y)
        
        # Step 2: Camera to world angles
        world_pan, world_tilt = self.camera_to_world_angles(camera_pan, camera_tilt)
        
        # Step 3: Apply parallax correction if needed
        if apply_parallax:
            turret_pan, turret_tilt = self.apply_parallax_correction(world_pan, world_tilt)
        else:
            turret_pan, turret_tilt = world_pan, world_tilt
        
        # Step 4: Constrain to turret limits
        final_pan, final_tilt = self.constrain_angles(turret_pan, turret_tilt)
        
        # Check if target is within limits
        is_valid = (turret_pan == final_pan and turret_tilt == final_tilt)
        
        return final_pan, final_tilt, is_valid
    
    def get_transformation_details(self, pixel_x: float, pixel_y: float) -> dict:
        """
        Get detailed transformation information for debugging.
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            
        Returns:
            Dictionary with all intermediate calculation results
        """
        # Step 1: Pixel to camera angles
        camera_pan, camera_tilt = self.pixel_to_camera_angles(pixel_x, pixel_y)
        
        # Step 2: Camera to world angles
        world_pan, world_tilt = self.camera_to_world_angles(camera_pan, camera_tilt)
        
        # Step 3: Apply parallax correction
        turret_pan_corrected, turret_tilt_corrected = self.apply_parallax_correction(
            world_pan, world_tilt
        )
        
        # Step 4: Constrain to turret limits
        final_pan, final_tilt = self.constrain_angles(turret_pan_corrected, turret_tilt_corrected)
        
        return {
            'input': {
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'camera_pan_position': self.camera.pan_position,
                'image_width': self.camera.image_width,
                'image_height': self.camera.image_height
            },
            'camera_relative': {
                'pan': camera_pan,
                'tilt': camera_tilt
            },
            'world_angles': {
                'pan': world_pan,
                'tilt': world_tilt
            },
            'parallax_corrected': {
                'pan': turret_pan_corrected,
                'tilt': turret_tilt_corrected
            },
            'final_angles': {
                'pan': final_pan,
                'tilt': final_tilt,
                'is_valid': (turret_pan_corrected == final_pan and 
                           turret_tilt_corrected == final_tilt)
            }
        }


def calculate_lead_angle(target_velocity: Tuple[float, float], 
                        projectile_speed: float,
                        target_distance: float) -> Tuple[float, float]:
    """
    Calculate lead angle for moving targets.
    
    Args:
        target_velocity: (vx, vy) target velocity in m/s
        projectile_speed: Speed of projectile in m/s
        target_distance: Distance to target in meters
        
    Returns:
        (lead_pan, lead_tilt) additional angles in degrees
    """
    # Time to target
    time_to_target = target_distance / projectile_speed
    
    # Predicted position change
    dx = target_velocity[0] * time_to_target
    dy = target_velocity[1] * time_to_target
    
    # Convert to angles
    lead_pan = math.degrees(math.atan2(dx, target_distance))
    lead_tilt = math.degrees(math.atan2(dy, target_distance))
    
    return lead_pan, lead_tilt