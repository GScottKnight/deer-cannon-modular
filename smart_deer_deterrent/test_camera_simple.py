#!/usr/bin/env python3
"""Simple camera test to isolate issues."""

import cv2
import time

print("Simple Camera Test")
print("==================\n")

# Test camera 0
camera_index = 0
print(f"Testing camera {camera_index}...")

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Failed to open camera {camera_index}")
    exit(1)

# Set buffer size to reduce lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera opened successfully!")
print("Press 'q' to quit\n")

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Display FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f}")
        
        # Show frame
        cv2.imshow('Camera Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    cap.release()
    cv2.destroyAllWindows()
    
print(f"\nTotal frames: {frame_count}")
print("Test complete")