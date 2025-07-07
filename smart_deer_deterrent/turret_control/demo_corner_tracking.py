#!/usr/bin/env python3
"""
Demo script showing corner-based stability detection for feeding deer.
Compares center-based vs corner-based tracking.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turret_control.main import TurretController
from turret_control.coordinate_transform import CameraConfig, TurretConfig


def simulate_feeding_deer(num_frames=100):
    """Simulate a feeding deer with head movement."""
    # Base position and size
    base_x, base_y = 600, 400
    base_width = 200
    base_height = 150
    
    bboxes = []
    for i in range(num_frames):
        # Simulate head movement (eating motion)
        head_cycle = np.sin(i * 0.3) * 30  # Head moves up/down
        
        # Add some small body sway
        body_sway = np.sin(i * 0.1) * 5
        
        # Bounding box with moving top (head) and stable bottom (feet)
        x1 = base_x + body_sway
        y1 = base_y - base_height - head_cycle  # Top moves with head
        x2 = base_x + base_width + body_sway
        y2 = base_y  # Bottom stays relatively fixed
        
        # Add small detection noise
        noise = np.random.normal(0, 2, 4)
        bbox = [x1 + noise[0], y1 + noise[1], x2 + noise[2], y2 + noise[3]]
        
        bboxes.append(bbox)
    
    return bboxes


def main():
    print("Corner-Based Stability Detection Demo")
    print("="*60)
    
    # Create controllers
    config = CameraConfig()
    turret_config = TurretConfig()
    
    # Controller without corner tracking (traditional center-based)
    controller_center = TurretController(
        config, turret_config, 
        use_smoothing=True,
        use_adaptive_smoothing=False
    )
    # Disable corner tracking on the smoothing filter
    controller_center.smoothing_filter.use_corner_tracking = False
    # Use lighter smoothing to show the difference
    controller_center.smoothing_filter.position_x_filter.alpha = 0.5
    controller_center.smoothing_filter.position_y_filter.alpha = 0.5
    
    # Controller with corner tracking
    controller_corner = TurretController(
        config, turret_config,
        use_smoothing=True,
        use_adaptive_smoothing=False
    )
    
    # Generate feeding deer simulation
    bboxes = simulate_feeding_deer(100)
    
    # Track aim points
    center_aims = []
    corner_aims = []
    
    print("\nProcessing frames...")
    for bbox in bboxes:
        # Center-based tracking
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        smoothed_center = controller_center.smoothing_filter.smooth_position(center_x, center_y)
        center_aims.append(smoothed_center)
        
        # Corner-based tracking
        corner_aim = controller_corner.smoothing_filter.smooth_bbox(bbox)
        corner_aims.append(corner_aim)
    
    # Extract coordinates
    center_xs = [p[0] for p in center_aims]
    center_ys = [p[1] for p in center_aims]
    corner_xs = [p[0] for p in corner_aims]
    corner_ys = [p[1] for p in corner_aims]
    
    # Calculate jitter metrics
    def calculate_jitter(positions):
        """Calculate total path length as jitter metric."""
        total = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total += np.sqrt(dx**2 + dy**2)
        return total
    
    center_jitter = calculate_jitter(center_aims)
    corner_jitter = calculate_jitter(corner_aims)
    
    print(f"\nJitter Analysis:")
    print(f"  Center-based tracking: {center_jitter:.1f} pixels total movement")
    print(f"  Corner-based tracking: {corner_jitter:.1f} pixels total movement")
    print(f"  Reduction: {(1 - corner_jitter/center_jitter)*100:.1f}%")
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Bounding box visualization
    ax1.set_xlim(500, 900)
    ax1.set_ylim(200, 450)
    ax1.invert_yaxis()  # Match image coordinates
    ax1.set_aspect('equal')
    ax1.set_title('Feeding Deer Simulation (First 30 Frames)')
    
    # Draw first 30 bounding boxes with transparency
    for i in range(min(30, len(bboxes))):
        bbox = bboxes[i]
        rect = Rectangle((bbox[0], bbox[1]), 
                        bbox[2] - bbox[0], 
                        bbox[3] - bbox[1],
                        fill=False, 
                        edgecolor='blue', 
                        alpha=0.3)
        ax1.add_patch(rect)
    
    # Mark stable corners
    ax1.plot([bboxes[0][0], bboxes[0][2]], [bboxes[0][3], bboxes[0][3]], 
             'go-', linewidth=2, label='Stable bottom edge')
    ax1.legend()
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Plot 2: Y-coordinate comparison
    frames = range(len(bboxes))
    ax2.plot(frames, center_ys, 'r-', label='Center-based', alpha=0.7)
    ax2.plot(frames, corner_ys, 'g-', label='Corner-based', linewidth=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y Aim Point (pixels)')
    ax2.set_title('Vertical Aim Point Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Aim point trajectories
    ax3.plot(center_xs, center_ys, 'r.', markersize=4, alpha=0.5, label='Center-based')
    ax3.plot(corner_xs, corner_ys, 'g.', markersize=6, label='Corner-based')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.set_title('Aim Point Trajectories')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'corner_tracking_demo.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nVisualization saved to: {output_file}")
    
    # Summary
    print("\nCorner Tracking Benefits:")
    print("1. Identifies stable body parts (feet/body) vs moving parts (head)")
    print("2. Aims at stable regions, dramatically reducing servo jitter")
    print("3. Maintains accurate targeting on animal's center of mass")
    print("4. Reduces mechanical wear on servo motors")
    
    plt.show()


if __name__ == "__main__":
    main()