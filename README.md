# Smart Deer Deterrent System

An intelligent wildlife management system that uses computer vision to detect deer and other animals, with integrated turret control for non-lethal deterrent deployment. The system prioritizes safety by never targeting animals when humans are present.

## 🎯 Features

- **Dual AI Model Detection**: Specialized deer detection + general object detection
- **Safety First**: Automatic detection of humans prevents any targeting when people are present
- **Live Camera Feed**: Real-time detection with multi-camera support
- **Mobile Interface**: Field deployment interface with joystick turret control
- **Cloud Integration**: Automatic upload to Azure with public viewing website
- **Smart Targeting**: Anatomical-aware smoothing for stable, accurate tracking
- **Detection Recording**: Saves 30-second clips of all detections with metadata

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenCV
- Camera connected to your system

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/smart-deer-deterrent.git
cd smart-deer-deterrent
```

2. Install dependencies:
```bash
cd smart_deer_deterrent
pip install -r requirements.txt
```

3. Download YOLO models:
   - Place `best.pt` (deer model) in `app/models/`
   - Place `yolov8n.pt` (general model) in `app/models/`

4. Run the application:
```bash
cd app
python main.py
```

5. Access the web interface at `http://localhost:5001`

## 📱 Mobile Field Deployment

Access the mobile interface at `http://YOUR_IP:5001/mobile` for:
- Live camera view optimized for mobile
- Manual turret control with on-screen joystick
- Quick access to recent detections
- System status monitoring

## ☁️ Azure Integration (Optional)

To enable cloud storage and public website:

1. Copy `.env.template` to `.env` and add your Azure credentials
2. Run the uploader service:
```bash
./scripts/start_uploader.sh
```
3. Deploy the public website from `public_website/` to your hosting service

See [AZURE_SETUP.md](smart_deer_deterrent/AZURE_SETUP.md) for detailed instructions.

## 🎮 Turret Control

The system includes sophisticated turret aiming calculations:
- Converts pixel coordinates to pan/tilt angles
- Handles camera pan offset
- Includes smoothing filters for stable tracking
- Configurable via `turret_control/turret_config.json`

**Note**: Servo hardware integration is pending. The system currently calculates angles but doesn't control physical servos.

## 🛠️ Configuration

### Model Settings (`config/model_config.json`)
- Detection confidence thresholds
- Model paths and versions
- Performance tuning options

### Turret Settings (`turret_control/turret_config.json`)
- Camera field of view
- Turret movement limits
- Physical offset calibration

## 📊 System Architecture

```
smart_deer_deterrent/
├── app/                    # Flask web application
│   ├── main.py            # Main application & routes
│   ├── models/            # YOLO model files
│   └── templates/         # Web interfaces
├── camera_detection/       # Detection engine
├── turret_control/        # Aiming calculations
├── azure_uploader.py      # Cloud upload service
└── public_website/        # Public viewing site
```

## 🔒 Safety Features

- **Human Detection Override**: Animals are marked as "Pets" when people are detected
- **10-Frame Persistence**: Continues safety mode for 10 frames after last human detection
- **Detection History**: Prevents false triggers from detection oscillation
- **Manual Emergency Stop**: Available in mobile interface

## 🚧 Development Status

### Completed ✅
- Core detection system with dual models
- Web interface (desktop & mobile)
- Azure cloud integration
- Turret angle calculations
- Detection recording and metadata

### In Progress 🔄
- Servo controller hardware integration
- Fire control mechanism
- Radar module integration

## 📈 Performance

- Processes 720p video at 15-30 FPS on modest hardware
- Configurable frame skipping for performance
- Cascaded inference option to reduce GPU load
- Optimized for edge devices (LattePanda, Jetson Nano)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## ⚖️ License

This project is for wildlife management and agricultural protection. Please ensure compliance with local wildlife protection laws and regulations.

## ⚠️ Disclaimer

This system is designed for non-lethal wildlife deterrence. Users are responsible for ensuring safe and legal deployment. Never use this system in a way that could harm humans or protected wildlife species.