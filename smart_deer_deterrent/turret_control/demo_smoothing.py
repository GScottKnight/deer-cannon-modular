#!/usr/bin/env python3
"""
Demo script showing the effect of smoothing on turret control.
This helps visualize how smoothing reduces servo jitter.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turret_control.main import TurretController
from turret_control.coordinate_transform import CameraConfig, TurretConfig


def generate_noisy_trajectory(num_points=100):
    """Generate a trajectory with realistic noise."""
    t = np.linspace(0, 4*np.pi, num_points)
    
    # Base trajectory: figure-8 pattern
    base_x = 640 + 200 * np.sin(t)
    base_y = 360 + 100 * np.sin(2*t)
    
    # Add detection noise
    noise_x = np.random.normal(0, 5, num_points)  # 5 pixel std dev
    noise_y = np.random.normal(0, 5, num_points)
    
    # Add occasional outliers (bad detections)
    for i in range(0, num_points, 20):
        if random.random() < 0.3:
            noise_x[i] += random.choice([-20, 20])
            noise_y[i] += random.choice([-20, 20])
    
    return base_x + noise_x, base_y + noise_y, base_x, base_y


def main():
    print("Turret Smoothing Demonstration")
    print("="*60)
    
    # Create controllers
    config = CameraConfig()
    turret_config = TurretConfig()
    
    controller_raw = TurretController(config, turret_config, use_smoothing=False)
    controller_smooth = TurretController(config, turret_config, use_smoothing=True)
    controller_adaptive = TurretController(config, turret_config, 
                                         use_smoothing=True, use_adaptive_smoothing=True)
    
    # Generate test trajectory
    noisy_x, noisy_y, true_x, true_y = generate_noisy_trajectory()
    
    # Process trajectory with each controller
    raw_pan, raw_tilt = [], []
    smooth_pan, smooth_tilt = [], []
    adaptive_pan, adaptive_tilt = [], []
    
    print("\nProcessing trajectory...")
    for i in range(len(noisy_x)):
        # Raw (no smoothing)
        pan, tilt, _ = controller_raw.transformer.transform(noisy_x[i], noisy_y[i])
        raw_pan.append(pan)
        raw_tilt.append(tilt)
        
        # Standard smoothing
        controller_smooth.aim_at_pixel(noisy_x[i], noisy_y[i])
        smooth_pan.append(controller_smooth.current_pan)
        smooth_tilt.append(controller_smooth.current_tilt)
        
        # Adaptive smoothing
        controller_adaptive.aim_at_pixel(noisy_x[i], noisy_y[i])
        adaptive_pan.append(controller_adaptive.current_pan)
        adaptive_tilt.append(controller_adaptive.current_tilt)
    
    # Calculate metrics
    def calculate_jitter(data):
        """Calculate jitter as RMS of successive differences."""
        diffs = np.diff(data)
        return np.sqrt(np.mean(diffs**2))
    
    print("\nJitter Analysis (RMS of successive differences):")
    print(f"  Raw Pan:      {calculate_jitter(raw_pan):.3f}°")
    print(f"  Smoothed Pan: {calculate_jitter(smooth_pan):.3f}°")
    print(f"  Adaptive Pan: {calculate_jitter(adaptive_pan):.3f}°")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Trajectory in image space
    ax1.plot(true_x, true_y, 'g-', label='True path', alpha=0.5, linewidth=2)
    ax1.scatter(noisy_x[::5], noisy_y[::5], c='red', s=10, alpha=0.5, label='Noisy detections')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title('Detection Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1280])
    ax1.set_ylim([0, 720])
    
    # Plot 2: Pan angles over time
    time_points = range(len(raw_pan))
    ax2.plot(time_points, raw_pan, 'r-', label='Raw', alpha=0.7)
    ax2.plot(time_points, smooth_pan, 'b-', label='Smoothed', linewidth=2)
    ax2.plot(time_points, adaptive_pan, 'g-', label='Adaptive', linewidth=2)
    ax2.set_xlabel('Time (frames)')
    ax2.set_ylabel('Pan Angle (degrees)')
    ax2.set_title('Pan Angle Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Zoom on pan angles to show jitter
    zoom_start, zoom_end = 40, 60
    ax3.plot(time_points[zoom_start:zoom_end], raw_pan[zoom_start:zoom_end], 
             'r.-', label='Raw', markersize=8)
    ax3.plot(time_points[zoom_start:zoom_end], smooth_pan[zoom_start:zoom_end], 
             'b.-', label='Smoothed', linewidth=2, markersize=8)
    ax3.set_xlabel('Time (frames)')
    ax3.set_ylabel('Pan Angle (degrees)')
    ax3.set_title('Pan Angle - Zoomed View (showing jitter)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Angular velocity (rate of change)
    raw_velocity = np.diff(raw_pan)
    smooth_velocity = np.diff(smooth_pan)
    
    ax4.plot(raw_velocity, 'r-', label='Raw', alpha=0.5)
    ax4.plot(smooth_velocity, 'b-', label='Smoothed', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (frames)')
    ax4.set_ylabel('Angular Velocity (deg/frame)')
    ax4.set_title('Pan Angular Velocity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'turret_smoothing_demo.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")
    
    # Additional analysis
    print("\nSmoothing Benefits:")
    print("1. Reduced servo wear from constant micro-movements")
    print("2. More stable tracking for physical systems")
    print("3. Better handling of detection noise and outliers")
    print("4. Configurable responsiveness vs smoothness trade-off")
    
    plt.show()


if __name__ == "__main__":
    main()