#!/usr/bin/env python3
import cv2
import time
import sys

def monitor_cameras():
    """Continuously monitor camera availability."""
    print("Camera Monitor - Press Ctrl+C to stop")
    print("=" * 60)
    
    last_cameras = set()
    
    while True:
        current_cameras = {}
        
        for i in range(5):  # Check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret:
                    status = "ACTIVE"
                    # Check if this might be ArduCam (usually has specific resolutions)
                    if width == 640 and height == 480:
                        status += " (Possibly ArduCam VGA)"
                    elif width == 1920 and height == 1080 and fps > 25:
                        status += " (Possibly ArduCam FHD)"
                    elif fps < 5:
                        status += " (Possibly iPhone/Continuity)"
                else:
                    status = "UNSTABLE"
                
                current_cameras[i] = {
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'status': status
                }
                
                cap.release()
        
        # Check for changes
        current_indices = set(current_cameras.keys())
        if current_indices != last_cameras:
            print(f"\n[{time.strftime('%H:%M:%S')}] Camera change detected!")
            
            # Show current cameras
            print("\nCurrent cameras:")
            for idx, info in sorted(current_cameras.items()):
                print(f"  Camera {idx}: {info['resolution']} @ {info['fps']:.1f} FPS - {info['status']}")
            
            # Show changes
            added = current_indices - last_cameras
            removed = last_cameras - current_indices
            
            if added:
                print(f"\n✅ NEW cameras: {sorted(added)}")
            if removed:
                print(f"\n❌ REMOVED cameras: {sorted(removed)}")
            
            print("-" * 60)
            last_cameras = current_indices
        
        time.sleep(2)  # Check every 2 seconds

if __name__ == "__main__":
    try:
        monitor_cameras()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")