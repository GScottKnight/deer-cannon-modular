#!/usr/bin/env python3
"""
Visual demonstration of the turret coordinate system.
Shows how camera pan and target position affect turret angles.
"""

import sys
import os
import numpy as np
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turret_control.coordinate_transform import (
    CoordinateTransformer, CameraConfig, TurretConfig
)


class CoordinateSystemDemo:
    """Visual demonstration of coordinate transformation."""
    
    def __init__(self):
        # Configuration
        self.camera_config = CameraConfig(
            horizontal_fov=60.0,
            vertical_fov=40.0,
            image_width=1280,
            image_height=720,
            pan_position=0.0
        )
        
        self.turret_config = TurretConfig(
            offset_x=0.0,
            offset_y=0.5,
            offset_z=0.0,
            target_distance=10.0
        )
        
        self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
        
        # Visualization settings
        self.viz_width = 1600
        self.viz_height = 900
        self.camera_pan = 0.0
        self.mouse_x = 640
        self.mouse_y = 360
        
    def draw_camera_view(self, img, x_offset=50, y_offset=50):
        """Draw camera view representation."""
        # Camera view rectangle
        cam_width = 400
        cam_height = 225
        cv2.rectangle(img, (x_offset, y_offset), 
                     (x_offset + cam_width, y_offset + cam_height), 
                     (255, 255, 255), 2)
        
        # Draw FOV lines
        center_x = x_offset + cam_width // 2
        center_y = y_offset + cam_height // 2
        
        # Draw grid
        for i in range(1, 4):
            x = x_offset + i * cam_width // 4
            cv2.line(img, (x, y_offset), (x, y_offset + cam_height), (128, 128, 128), 1)
        for i in range(1, 4):
            y = y_offset + i * cam_height // 4
            cv2.line(img, (x_offset, y), (x_offset + cam_width, y), (128, 128, 128), 1)
        
        # Draw center crosshair
        cv2.line(img, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
        cv2.line(img, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)
        
        # Draw target position
        target_x = x_offset + int(self.mouse_x * cam_width / self.camera_config.image_width)
        target_y = y_offset + int(self.mouse_y * cam_height / self.camera_config.image_height)
        cv2.circle(img, (target_x, target_y), 8, (0, 0, 255), -1)
        cv2.circle(img, (target_x, target_y), 12, (0, 0, 255), 2)
        
        # Labels
        cv2.putText(img, "Camera View", (x_offset, y_offset - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Camera Pan: {self.camera_pan:.1f}°", 
                   (x_offset, y_offset + cam_height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return cam_width, cam_height
    
    def draw_top_view(self, img, x_offset=500, y_offset=50):
        """Draw top-down view of camera and turret."""
        view_size = 300
        center_x = x_offset + view_size // 2
        center_y = y_offset + view_size // 2
        
        # Draw coordinate system
        cv2.rectangle(img, (x_offset, y_offset), 
                     (x_offset + view_size, y_offset + view_size), 
                     (255, 255, 255), 2)
        
        # Draw grid
        for i in range(1, 6):
            pos = x_offset + i * view_size // 6
            cv2.line(img, (pos, y_offset), (pos, y_offset + view_size), (64, 64, 64), 1)
            cv2.line(img, (x_offset, pos), (x_offset + view_size, pos), (64, 64, 64), 1)
        
        # Draw camera position and FOV
        cam_x = center_x
        cam_y = center_y
        cv2.circle(img, (cam_x, cam_y), 8, (0, 255, 0), -1)
        
        # Draw camera FOV cone
        fov_length = 100
        fov_half = self.camera_config.horizontal_fov / 2
        left_angle = np.radians(self.camera_pan - fov_half)
        right_angle = np.radians(self.camera_pan + fov_half)
        center_angle = np.radians(self.camera_pan)
        
        # FOV lines
        left_x = int(cam_x + fov_length * np.sin(left_angle))
        left_y = int(cam_y - fov_length * np.cos(left_angle))
        right_x = int(cam_x + fov_length * np.sin(right_angle))
        right_y = int(cam_y - fov_length * np.cos(right_angle))
        center_x2 = int(cam_x + fov_length * np.sin(center_angle))
        center_y2 = int(cam_y - fov_length * np.cos(center_angle))
        
        cv2.line(img, (cam_x, cam_y), (left_x, left_y), (0, 255, 0), 1)
        cv2.line(img, (cam_x, cam_y), (right_x, right_y), (0, 255, 0), 1)
        cv2.line(img, (cam_x, cam_y), (center_x2, center_y2), (0, 255, 0), 2)
        
        # Draw turret position
        turret_x = cam_x + int(self.turret_config.offset_x * 20)
        turret_y = cam_y + int(self.turret_config.offset_z * 20)
        cv2.circle(img, (turret_x, turret_y), 8, (255, 0, 0), -1)
        
        # Get turret angles
        details = self.transformer.get_transformation_details(self.mouse_x, self.mouse_y)
        turret_pan = details['final_angles']['pan']
        
        # Draw turret aim line
        aim_length = 120
        aim_angle = np.radians(turret_pan)
        aim_x = int(turret_x + aim_length * np.sin(aim_angle))
        aim_y = int(turret_y - aim_length * np.cos(aim_angle))
        cv2.line(img, (turret_x, turret_y), (aim_x, aim_y), (255, 0, 0), 2)
        
        # Labels
        cv2.putText(img, "Top View", (x_offset, y_offset - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "Camera", (cam_x - 30, cam_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, "Turret", (turret_x - 30, turret_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    def draw_info_panel(self, img, x_offset=900, y_offset=50):
        """Draw information panel with calculations."""
        panel_width = 650
        panel_height = 400
        
        # Panel background
        cv2.rectangle(img, (x_offset, y_offset), 
                     (x_offset + panel_width, y_offset + panel_height), 
                     (255, 255, 255), 2)
        
        # Get transformation details
        details = self.transformer.get_transformation_details(self.mouse_x, self.mouse_y)
        
        # Title
        cv2.putText(img, "Coordinate Transformation", (x_offset + 10, y_offset + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display information
        y_pos = y_offset + 70
        line_height = 30
        
        info_lines = [
            f"Mouse Position: ({self.mouse_x}, {self.mouse_y})",
            f"Camera Pan: {self.camera_pan:.1f}°",
            "",
            "Camera-Relative Angles:",
            f"  Pan: {details['camera_relative']['pan']:.2f}°",
            f"  Tilt: {details['camera_relative']['tilt']:.2f}°",
            "",
            "World Angles:",
            f"  Pan: {details['world_angles']['pan']:.2f}°",
            f"  Tilt: {details['world_angles']['tilt']:.2f}°",
            "",
            "Turret Angles (with parallax):",
            f"  Pan: {details['final_angles']['pan']:.2f}°",
            f"  Tilt: {details['final_angles']['tilt']:.2f}°",
            f"  Valid: {'Yes' if details['final_angles']['is_valid'] else 'No (outside limits)'}"
        ]
        
        for line in info_lines:
            if line:
                cv2.putText(img, line, (x_offset + 20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += line_height
    
    def draw_controls(self, img):
        """Draw control instructions."""
        y_offset = 500
        x_offset = 50
        
        cv2.putText(img, "Controls:", (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        controls = [
            "Mouse: Move target in camera view",
            "Left/Right Arrow: Pan camera",
            "Up/Down Arrow: Change target distance",
            "R: Reset to defaults",
            "ESC: Exit"
        ]
        
        y_pos = y_offset + 40
        for control in controls:
            cv2.putText(img, control, (x_offset + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 30
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        # Map mouse position to camera view area
        cam_x_start = 50
        cam_y_start = 50
        cam_width = 400
        cam_height = 225
        
        if cam_x_start <= x <= cam_x_start + cam_width and \
           cam_y_start <= y <= cam_y_start + cam_height:
            # Convert to image coordinates
            rel_x = x - cam_x_start
            rel_y = y - cam_y_start
            self.mouse_x = int(rel_x * self.camera_config.image_width / cam_width)
            self.mouse_y = int(rel_y * self.camera_config.image_height / cam_height)
    
    def run(self):
        """Run the demonstration."""
        cv2.namedWindow('Turret Coordinate System Demo')
        cv2.setMouseCallback('Turret Coordinate System Demo', self.mouse_callback)
        
        print("Turret Coordinate System Demonstration")
        print("=====================================")
        print("Controls:")
        print("  Mouse: Move target in camera view")
        print("  Left/Right Arrow: Pan camera")
        print("  Up/Down Arrow: Change target distance")
        print("  R: Reset to defaults")
        print("  ESC: Exit")
        
        while True:
            # Create blank image
            img = np.zeros((self.viz_height, self.viz_width, 3), dtype=np.uint8)
            
            # Draw components
            self.draw_camera_view(img)
            self.draw_top_view(img)
            self.draw_info_panel(img)
            self.draw_controls(img)
            
            # Show image
            cv2.imshow('Turret Coordinate System Demo', img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 81 or key == 2:  # Left arrow
                self.camera_pan = max(-90, self.camera_pan - 5)
                self.camera_config.pan_position = self.camera_pan
                self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
            elif key == 83 or key == 3:  # Right arrow
                self.camera_pan = min(90, self.camera_pan + 5)
                self.camera_config.pan_position = self.camera_pan
                self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
            elif key == 82 or key == 0:  # Up arrow
                self.turret_config.target_distance = min(50, self.turret_config.target_distance + 2)
                self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
            elif key == 84 or key == 1:  # Down arrow
                self.turret_config.target_distance = max(2, self.turret_config.target_distance - 2)
                self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
            elif key == ord('r') or key == ord('R'):  # Reset
                self.camera_pan = 0.0
                self.camera_config.pan_position = 0.0
                self.turret_config.target_distance = 10.0
                self.mouse_x = 640
                self.mouse_y = 360
                self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    demo = CoordinateSystemDemo()
    demo.run()