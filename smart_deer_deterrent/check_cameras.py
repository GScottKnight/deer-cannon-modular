#!/usr/bin/env python3
"""Check available cameras and their properties."""

import cv2
import platform

print(f"Platform: {platform.system()}")
print(f"OpenCV version: {cv2.__version__}")
print("\nChecking cameras...\n")

# Try different camera backends for macOS
if platform.system() == "Darwin":
    backends = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation"),
        (cv2.CAP_ANY, "Default")
    ]
else:
    backends = [(cv2.CAP_ANY, "Default")]

for backend_id, backend_name in backends:
    print(f"\nTrying {backend_name} backend:")
    for i in range(5):
        # Try to open camera with specific backend
        cap = cv2.VideoCapture(i, backend_id)
        
        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to read a frame
            ret, frame = cap.read()
            
            if ret:
                print(f"  Camera {i}: {width}x{height} @ {fps} FPS - Working ✓")
            else:
                print(f"  Camera {i}: Detected but can't read frames")
            
            cap.release()
        else:
            if i < 2:  # Only show first 2 attempts to reduce noise
                print(f"  Camera {i}: Not available")

print("\nFor macOS, you may need to grant camera permissions to Terminal/Python")
print("System Preferences → Privacy & Security → Camera")