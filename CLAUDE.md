# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Smart Deer Deterrent System built with Python/Flask that uses computer vision (YOLO) to detect deer and other animals, with the capability to control a turret-based deterrent mechanism.

## Development Commands

### Running the Application
```bash
# Install dependencies (requirements.txt is empty, so install manually)
pip install flask opencv-python ultralytics werkzeug

# Run the Flask web application
cd smart_deer_deterrent/app
python main.py
```

The application runs on `http://0.0.0.0:5001` and provides a web interface for:
- Uploading images/videos for animal detection
- Live camera feed with real-time detection
- Camera selection for multi-camera setups

### Testing
No test framework is currently configured. When implementing tests, consider using pytest.

### Linting
No linting configuration exists. Consider using flake8 or pylint for Python code quality.

## Architecture Overview

### Module Structure
```
smart_deer_deterrent/
├── app/                    # Flask web application
│   ├── main.py            # Main Flask app with routes
│   ├── models/            # YOLO model files (.pt)
│   ├── static/            # Processed videos
│   ├── templates/         # HTML templates
│   └── uploads/           # User uploaded files
├── camera_detection/       # Core detection logic
│   └── detector.py        # YOLO inference and target selection
├── turret_control/        # Turret aiming calculations
│   └── main.py           # TurretController class
├── servo_controller/      # [Empty] Servo motor control
├── fire_control/          # [Empty] Firing mechanism
└── radar_module/          # [Empty] Radar integration
```

### Key Components

1. **Detection System** (`camera_detection/detector.py`):
   - Uses two YOLO models: `deer_model.pt` (specialized) and `yolov8n.pt` (general)
   - Implements safety logic: won't target animals when humans are present
   - Filters overlapping detections using IoU calculations
   - Returns target coordinates for turret aiming

2. **Web Interface** (`app/main.py`):
   - Flask routes for file upload, camera feed, and detection results
   - Video streaming with real-time detection overlay
   - Supports both image and video file processing

3. **Turret Control** (`turret_control/main.py`):
   - Converts pixel coordinates to pan/tilt angles
   - Uses camera field-of-view for angle calculations
   - Designed to interface with ServoController (not yet implemented)

### Important Design Decisions

- **Safety First**: System marks animals as "Pets" when humans are detected
- **Dual Model Approach**: Combines specialized deer detection with general object detection
- **Modular Architecture**: Clear separation between detection, control, and actuation
- **Web-Based Interface**: Accessible from any device on the network

### Missing Implementations

- `math_utils` module (referenced but not found)
- `ServoController` class implementation
- Fire control logic
- Radar integration
- Requirements.txt content
- Unit tests and CI/CD pipeline