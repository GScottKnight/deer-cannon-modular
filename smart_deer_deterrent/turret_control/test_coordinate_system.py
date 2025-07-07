#!/usr/bin/env python3
"""
Test harness for turret coordinate transformation system.

This script allows testing various scenarios for camera-to-turret coordinate
transformations, including edge cases and calibration verification.
"""

import sys
import os
import math
import argparse
import time
import numpy as np
from typing import List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turret_control.coordinate_transform import (
    CoordinateTransformer, CameraConfig, TurretConfig
)
from turret_control.main import TurretController


class TestHarness:
    """Test harness for coordinate transformation."""
    
    def __init__(self):
        # Default configurations
        self.camera_config = CameraConfig(
            horizontal_fov=60.0,
            vertical_fov=40.0,
            image_width=1280,
            image_height=720,
            pan_position=0.0
        )
        
        self.turret_config = TurretConfig(
            offset_x=0.0,    # 0cm left/right
            offset_y=0.5,    # 50cm below camera
            offset_z=0.0,    # 0cm forward/back
            pan_min=-90.0,
            pan_max=90.0,
            tilt_min=-30.0,
            tilt_max=45.0,
            target_distance=10.0  # 10 meters
        )
        
        self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
    
    def run_test_case(self, name: str, pixel_x: float, pixel_y: float, 
                      camera_pan: float = 0.0) -> None:
        """Run a single test case and print results."""
        print(f"\n{'='*60}")
        print(f"Test Case: {name}")
        print(f"{'='*60}")
        
        # Update camera pan position
        self.camera_config.pan_position = camera_pan
        self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
        
        # Get transformation details
        details = self.transformer.get_transformation_details(pixel_x, pixel_y)
        
        # Print results
        print(f"Input:")
        print(f"  Camera Pan Position: {camera_pan:.1f}°")
        print(f"  Pixel Coordinates: ({pixel_x:.0f}, {pixel_y:.0f})")
        print(f"  Image Center: ({self.camera_config.image_width/2:.0f}, "
              f"{self.camera_config.image_height/2:.0f})")
        
        print(f"\nCamera-Relative Angles:")
        print(f"  Pan:  {details['camera_relative']['pan']:7.2f}°")
        print(f"  Tilt: {details['camera_relative']['tilt']:7.2f}°")
        
        print(f"\nWorld Angles:")
        print(f"  Pan:  {details['world_angles']['pan']:7.2f}°")
        print(f"  Tilt: {details['world_angles']['tilt']:7.2f}°")
        
        print(f"\nParallax-Corrected Turret Angles:")
        print(f"  Pan:  {details['parallax_corrected']['pan']:7.2f}°")
        print(f"  Tilt: {details['parallax_corrected']['tilt']:7.2f}°")
        
        print(f"\nFinal Turret Angles (after limits):")
        print(f"  Pan:  {details['final_angles']['pan']:7.2f}°")
        print(f"  Tilt: {details['final_angles']['tilt']:7.2f}°")
        print(f"  Valid: {'✓' if details['final_angles']['is_valid'] else '✗ (outside limits)'}")
    
    def run_standard_tests(self) -> None:
        """Run standard test cases."""
        print("\nRUNNING STANDARD TEST CASES")
        print("="*80)
        
        # Test 1: Target at image center, camera at 0°
        self.run_test_case(
            "Center target, camera at 0°",
            pixel_x=640, pixel_y=360,
            camera_pan=0.0
        )
        
        # Test 2: Target at right edge, camera at 0°
        self.run_test_case(
            "Right edge target, camera at 0°",
            pixel_x=1280, pixel_y=360,
            camera_pan=0.0
        )
        
        # Test 3: Target at center, camera panned 45° right
        self.run_test_case(
            "Center target, camera at 45°",
            pixel_x=640, pixel_y=360,
            camera_pan=45.0
        )
        
        # Test 4: Target at left edge, camera panned 45° right
        self.run_test_case(
            "Left edge target, camera at 45°",
            pixel_x=0, pixel_y=360,
            camera_pan=45.0
        )
        
        # Test 5: Target at top-left corner
        self.run_test_case(
            "Top-left corner target, camera at 0°",
            pixel_x=0, pixel_y=0,
            camera_pan=0.0
        )
        
        # Test 6: Target at bottom-right corner
        self.run_test_case(
            "Bottom-right corner target, camera at 0°",
            pixel_x=1280, pixel_y=720,
            camera_pan=0.0
        )
        
        # Test 7: Extreme camera pan with centered target
        self.run_test_case(
            "Center target, camera at -60°",
            pixel_x=640, pixel_y=360,
            camera_pan=-60.0
        )
    
    def run_edge_case_tests(self) -> None:
        """Run edge case tests."""
        print("\n\nRUNNING EDGE CASE TESTS")
        print("="*80)
        
        # Test with extreme turret offset
        print("\nTesting with large turret offset (2m horizontal)...")
        self.turret_config.offset_x = 2.0  # 2 meters to the right
        self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
        
        self.run_test_case(
            "Center target with 2m horizontal offset",
            pixel_x=640, pixel_y=360,
            camera_pan=0.0
        )
        
        # Reset offset
        self.turret_config.offset_x = 0.0
        
        # Test with very close target
        print("\nTesting with very close target (2m)...")
        self.turret_config.target_distance = 2.0
        self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
        
        self.run_test_case(
            "Center target at 2m distance",
            pixel_x=640, pixel_y=360,
            camera_pan=0.0
        )
        
        # Reset distance
        self.turret_config.target_distance = 10.0
    
    def run_calibration_test(self) -> None:
        """Run calibration verification test."""
        print("\n\nRUNNING CALIBRATION TEST")
        print("="*80)
        
        print("\nThis test helps verify physical alignment:")
        print("1. Aim camera at a known target")
        print("2. Note the pixel coordinates of the target")
        print("3. Measure actual angles with a protractor")
        print("4. Compare with calculated values\n")
        
        # Simulate calibration points
        calibration_points = [
            (640, 360, "Center point"),
            (320, 360, "Left quarter"),
            (960, 360, "Right quarter"),
            (640, 180, "Top center"),
            (640, 540, "Bottom center")
        ]
        
        print("Calibration Reference Points:")
        print("-" * 60)
        print(f"{'Description':<20} {'Pixel':<15} {'Expected Pan':<15} {'Expected Tilt':<15}")
        print("-" * 60)
        
        for px, py, desc in calibration_points:
            pan, tilt, _ = self.transformer.transform(px, py, apply_parallax=False)
            print(f"{desc:<20} ({px:4d},{py:4d})     {pan:7.2f}°        {tilt:7.2f}°")
    
    def run_interactive_mode(self) -> None:
        """Run interactive test mode."""
        print("\n\nINTERACTIVE TEST MODE")
        print("="*80)
        print("Enter pixel coordinates and camera pan to test.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                # Get input
                user_input = input("Enter: pixel_x pixel_y camera_pan (or 'quit'): ")
                
                if user_input.lower() == 'quit':
                    break
                
                # Parse input
                parts = user_input.split()
                if len(parts) != 3:
                    print("Error: Please enter three numbers separated by spaces")
                    continue
                
                pixel_x = float(parts[0])
                pixel_y = float(parts[1])
                camera_pan = float(parts[2])
                
                # Validate input
                if not (0 <= pixel_x <= self.camera_config.image_width):
                    print(f"Error: pixel_x must be between 0 and {self.camera_config.image_width}")
                    continue
                
                if not (0 <= pixel_y <= self.camera_config.image_height):
                    print(f"Error: pixel_y must be between 0 and {self.camera_config.image_height}")
                    continue
                
                # Run test
                self.run_test_case(
                    "Interactive Test",
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    camera_pan=camera_pan
                )
                
            except ValueError:
                print("Error: Please enter valid numbers")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
    
    def export_csv_data(self, filename: str = "turret_calibration_data.csv") -> None:
        """Export calibration data to CSV for analysis."""
        print(f"\nExporting calibration data to {filename}...")
        
        with open(filename, 'w') as f:
            # Write header
            f.write("camera_pan,pixel_x,pixel_y,camera_rel_pan,camera_rel_tilt,"
                   "world_pan,world_tilt,turret_pan,turret_tilt,is_valid\n")
            
            # Generate test points
            camera_pans = [-60, -30, 0, 30, 60]
            pixel_xs = [0, 320, 640, 960, 1280]
            pixel_ys = [0, 180, 360, 540, 720]
            
            for cam_pan in camera_pans:
                self.camera_config.pan_position = cam_pan
                self.transformer = CoordinateTransformer(self.camera_config, self.turret_config)
                
                for px in pixel_xs:
                    for py in pixel_ys:
                        details = self.transformer.get_transformation_details(px, py)
                        
                        f.write(f"{cam_pan},{px},{py},"
                               f"{details['camera_relative']['pan']:.2f},"
                               f"{details['camera_relative']['tilt']:.2f},"
                               f"{details['world_angles']['pan']:.2f},"
                               f"{details['world_angles']['tilt']:.2f},"
                               f"{details['parallax_corrected']['pan']:.2f},"
                               f"{details['parallax_corrected']['tilt']:.2f},"
                               f"{details['final_angles']['is_valid']}\n")
        
        print(f"Data exported successfully!")
    
    def run_smoothing_test(self) -> None:
        """Test smoothing functionality for servo control."""
        print("\n\nRUNNING SMOOTHING TEST")
        print("="*80)
        
        import random
        import time
        
        # Create controllers with different smoothing settings
        print("\nCreating controllers with different smoothing settings...")
        controller_no_smooth = TurretController(
            camera_config=self.camera_config,
            turret_config=self.turret_config,
            use_smoothing=False
        )
        controller_smooth = TurretController(
            camera_config=self.camera_config,
            turret_config=self.turret_config,
            use_smoothing=True
        )
        controller_adaptive = TurretController(
            camera_config=self.camera_config,
            turret_config=self.turret_config,
            use_smoothing=True,
            use_adaptive_smoothing=True
        )
        
        # Test 1: Noisy stationary target
        print("\n--- Test 1: Noisy Stationary Target ---")
        print("Simulating detection jitter around a fixed point...")
        base_x, base_y = 640, 360
        
        print(f"\n{'Frame':<8} {'Raw X':<8} {'Raw Y':<8} {'No Smooth Pan':<14} {'Smooth Pan':<14} {'Adaptive Pan':<14}")
        print("-" * 80)
        
        for i in range(10):
            # Add small random noise
            noise_x = random.uniform(-5, 5)
            noise_y = random.uniform(-5, 5)
            x = base_x + noise_x
            y = base_y + noise_y
            
            # Calculate angles with each controller
            pan1, tilt1, _ = controller_no_smooth.transformer.transform(x, y)
            
            # For smoothed controllers, we need to use aim_at_pixel
            controller_smooth.aim_at_pixel(x, y)
            controller_adaptive.aim_at_pixel(x, y)
            
            print(f"{i+1:<8} {x:<8.1f} {y:<8.1f} {pan1:<14.2f} "
                  f"{controller_smooth.current_pan:<14.2f} {controller_adaptive.current_pan:<14.2f}")
            
            time.sleep(0.05)  # Simulate frame rate
        
        # Test 2: Fast moving target
        print("\n\n--- Test 2: Fast Moving Target ---")
        print("Simulating fast horizontal movement...")
        
        print(f"\n{'Frame':<8} {'X':<8} {'Y':<8} {'No Smooth Pan':<14} {'Smooth Pan':<14} {'Adaptive Pan':<14}")
        print("-" * 80)
        
        # Reset smoothing filters
        controller_smooth.reset_smoothing()
        controller_adaptive.reset_smoothing()
        
        for i in range(15):
            # Target moves quickly from left to right
            x = 100 + i * 80
            y = 360 + random.uniform(-2, 2)  # Small vertical noise
            
            # Calculate angles
            pan1, tilt1, _ = controller_no_smooth.transformer.transform(x, y)
            controller_smooth.aim_at_pixel(x, y)
            controller_adaptive.aim_at_pixel(x, y)
            
            print(f"{i+1:<8} {x:<8.1f} {y:<8.1f} {pan1:<14.2f} "
                  f"{controller_smooth.current_pan:<14.2f} {controller_adaptive.current_pan:<14.2f}")
            
            time.sleep(0.05)
        
        # Test 3: Smoothing parameters adjustment
        print("\n\n--- Test 3: Smoothing Parameter Effects ---")
        print("Testing different smoothing alpha values...")
        
        # Reset and test with different smoothing factors
        controller_smooth.reset_smoothing()
        
        smoothing_factors = [0.1, 0.3, 0.5, 0.8]
        
        for alpha in smoothing_factors:
            controller_smooth.set_smoothing_params(position_alpha=alpha, angle_alpha=alpha)
            print(f"\n  Alpha = {alpha}:")
            
            # Quick movement test
            positions = [320, 640, 960, 640, 320]
            for x in positions:
                controller_smooth.aim_at_pixel(x, 360)
                print(f"    Target X={x:<4} -> Smoothed Pan={controller_smooth.current_pan:.2f}°")
                time.sleep(0.05)
        
        print("\n\nSmoothing Analysis:")
        print("- Lower alpha (0.1) = More smoothing, slower response")
        print("- Higher alpha (0.8) = Less smoothing, faster response")
        print("- Adaptive smoothing adjusts alpha based on movement speed")
    
    def run_corner_tracking_test(self) -> None:
        """Test corner-based stability detection."""
        print("\n\nRUNNING CORNER TRACKING TEST")
        print("="*80)
        
        # Create controller with corner tracking
        controller = TurretController(
            camera_config=self.camera_config,
            turret_config=self.turret_config,
            use_smoothing=True
        )
        
        print("\n--- Test 1: Feeding Deer Simulation ---")
        print("Simulating deer with head moving up/down while body stays still...")
        
        # Base position and size
        base_x, base_y = 600, 400
        base_width, base_height = 200, 150
        
        print(f"\n{'Frame':<8} {'Box Height':<12} {'Center Y':<12} {'Stable Corners':<30} {'Aim Point':<20}")
        print("-" * 90)
        
        for i in range(20):
            # Simulate head movement by changing top of bbox
            head_movement = 30 * np.sin(i * 0.5)  # Head moves up and down
            
            # Bounding box: bottom stays fixed (feet), top moves (head)
            x1 = base_x
            y1 = base_y - base_height - head_movement  # Top moves
            x2 = base_x + base_width
            y2 = base_y  # Bottom fixed
            
            bbox = [x1, y1, x2, y2]
            height = y2 - y1
            center_y = (y1 + y2) / 2
            
            # Aim at target
            controller.aim_at_target(bbox)
            
            # Get corner info
            info = controller.get_turret_info()
            corner_info = info.get('corner_tracking', {})
            stable_corners = corner_info.get('stable_corners', [])
            
            # Get actual aim point
            if controller.smoothing_filter:
                aim_x, aim_y = controller.smoothing_filter.smooth_bbox(bbox)
            else:
                aim_x = (x1 + x2) / 2
                aim_y = center_y
            
            print(f"{i+1:<8} {height:<12.1f} {center_y:<12.1f} "
                  f"{', '.join(stable_corners[:2]) if stable_corners else 'None':<30} "
                  f"({aim_x:.0f}, {aim_y:.0f})")
            
            time.sleep(0.05)
        
        print("\n\n--- Test 2: Corner Stability Scores ---")
        if corner_info:
            print("\nFinal stability scores (lower = more stable):")
            for corner, score in corner_info.get('stability_scores', {}).items():
                print(f"  {corner}: {score:.2f}")
            
            if corner_info.get('most_stable'):
                print(f"\nMost stable corner: {corner_info['most_stable']}")
        
        print("\n\nCorner Tracking Benefits:")
        print("- Automatically detects stable body parts")
        print("- Aims at stable regions instead of jittery center")
        print("- Reduces servo wear from unnecessary movements")
        print("- Maintains accurate targeting on the animal's mass")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test harness for turret coordinate transformation"
    )
    parser.add_argument(
        '--mode', 
        choices=['standard', 'edge', 'calibration', 'smoothing', 'corner', 'interactive', 'all'],
        default='all',
        help='Test mode to run'
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Export calibration data to CSV'
    )
    
    args = parser.parse_args()
    
    # Create test harness
    harness = TestHarness()
    
    # Print configuration
    print("TURRET COORDINATE TRANSFORMATION TEST HARNESS")
    print("="*80)
    print(f"Camera Configuration:")
    print(f"  FOV: {harness.camera_config.horizontal_fov}° x "
          f"{harness.camera_config.vertical_fov}°")
    print(f"  Resolution: {harness.camera_config.image_width} x "
          f"{harness.camera_config.image_height}")
    print(f"\nTurret Configuration:")
    print(f"  Offset: X={harness.turret_config.offset_x}m, "
          f"Y={harness.turret_config.offset_y}m, "
          f"Z={harness.turret_config.offset_z}m")
    print(f"  Limits: Pan=[{harness.turret_config.pan_min}°, "
          f"{harness.turret_config.pan_max}°], "
          f"Tilt=[{harness.turret_config.tilt_min}°, "
          f"{harness.turret_config.tilt_max}°]")
    print(f"  Target Distance: {harness.turret_config.target_distance}m")
    
    # Run tests based on mode
    if args.mode in ['standard', 'all']:
        harness.run_standard_tests()
    
    if args.mode in ['edge', 'all']:
        harness.run_edge_case_tests()
    
    if args.mode in ['calibration', 'all']:
        harness.run_calibration_test()
    
    if args.mode in ['smoothing', 'all']:
        harness.run_smoothing_test()
    
    if args.mode in ['corner', 'all']:
        harness.run_corner_tracking_test()
    
    if args.mode == 'interactive':
        harness.run_interactive_mode()
    
    # Export CSV if requested
    if args.export_csv:
        harness.export_csv_data()
    
    print("\n\nTest harness complete!")


if __name__ == "__main__":
    main()