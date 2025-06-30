import sys
import os

# Adjust the path to include the parent directory (smart_deer_deterrent)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from servo_controller.main import ServoController

class TurretController:
    def __init__(self, frame_width, frame_height, horizontal_fov=60, vertical_fov=40):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.servo_controller = ServoController()

        # Degrees per pixel
        self.deg_per_pixel_x = horizontal_fov / frame_width
        self.deg_per_pixel_y = vertical_fov / frame_height

    def aim_at_target(self, target_box):
        """
        Calculates the required pan and tilt adjustments to aim at the center of the target.
        """
        # Calculate the center of the bounding box
        target_center_x = (target_box[0] + target_box[2]) / 2
        target_center_y = (target_box[1] + target_box[3]) / 2

        # Calculate the center of the frame
        frame_center_x = self.frame_width / 2
        frame_center_y = self.frame_height / 2

        # Calculate the offset from the center
        pan_offset = target_center_x - frame_center_x
        tilt_offset = target_center_y - frame_center_y

        # Convert pixel offset to degrees
        pan_angle = pan_offset * self.deg_per_pixel_x
        tilt_angle = -(tilt_offset * self.deg_per_pixel_y)  # Invert y-axis

        print(f"Targeting: pan={pan_angle:.2f}°, tilt={tilt_angle:.2f}°")
        self.servo_controller.aim_at(pan_angle, tilt_angle)
