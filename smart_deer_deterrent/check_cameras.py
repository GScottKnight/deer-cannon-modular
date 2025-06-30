#!/usr/bin/env python3
import cv2
import time

def check_cameras():
    """Check all available cameras and their properties."""
    print("Checking for available cameras...")
    print("-" * 50)
    
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to read a frame to confirm it's working
            ret, frame = cap.read()
            
            if ret:
                print(f"Camera {i}: ACTIVE")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")
                available_cameras.append(i)
            else:
                print(f"Camera {i}: Detected but couldn't read frame")
            
            cap.release()
        
    print("-" * 50)
    print(f"Total active cameras found: {len(available_cameras)}")
    print(f"Camera indices: {available_cameras}")
    
    return available_cameras

if __name__ == "__main__":
    print("Initial camera check:")
    initial_cameras = check_cameras()
    
    print("\n\nPlease plug in your ArduCam now...")
    print("Waiting 5 seconds...")
    time.sleep(5)
    
    print("\n\nChecking again for new cameras:")
    final_cameras = check_cameras()
    
    new_cameras = set(final_cameras) - set(initial_cameras)
    if new_cameras:
        print(f"\n\nNEW CAMERA DETECTED at index: {list(new_cameras)}")
    else:
        print("\n\nNo new cameras detected. Try unplugging and replugging the ArduCam.")