import base64
import os
import subprocess
import cv2
from flask import Flask, render_template, request, jsonify, Response, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import sys
import json
from datetime import datetime
import shutil
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.model_manager import model_manager
from shared.math_utils import calculate_iou
from shared.detection_tracker import DetectionTracker
from logger_config import logger_config
from system_monitor import system_monitor

# Initialize loggers
app_logger = logger_config.get_app_logger()
detection_logger = logger_config.get_detection_logger()
camera_logger = logger_config.get_camera_logger()
system_logger = logger_config.get_system_logger()

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(_script_dir, 'uploads')
STATIC_FOLDER = os.path.join(_script_dir, 'static')
DETECTIONS_FOLDER = os.path.join(_script_dir, 'detections')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTIONS_FOLDER'] = DETECTIONS_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

# --- Model Optimization Settings ---
MODEL_CONFIG = {
    'imgsz': 640,  # Input image size (can be reduced to 416 or 320 for faster inference)
    'device': 'cpu',  # 'cpu', 'cuda', 'mps' (for Mac M1/M2)
    'half': False,  # Use FP16 precision (only works on CUDA)
    'max_det': 100,  # Maximum detections per image
    'conf_threshold': 0.5,  # Confidence threshold (standard level)
    'iou_threshold': 0.45,  # IoU threshold for NMS
    'batch_size': 1,  # Batch size for inference
    'use_cascaded': False,  # Temporarily disable cascaded to ensure detections work
    'frame_skip': 1,  # Process every Nth frame (1 = process all frames)
    'detection_persistence': 10,  # Number of frames to persist detections when not detected
}

# --- Detection Video Settings ---
DETECTION_VIDEO_CONFIG = {
    'save_detections': True,  # Enable saving detection video clips
    'clip_duration_seconds': 30,  # Duration of detection clips in seconds
    'pre_detection_seconds': 5,  # Seconds to include before detection
    'post_detection_seconds': 5,  # Seconds to include after detection
    'min_confidence': 0.5,  # Minimum confidence to trigger detection save
    'save_metadata': True,  # Save JSON metadata with each detection
    'max_storage_days': 30,  # Days to keep detection videos
}

# --- Model Loading with Optimization ---
DEER_MODEL_PATH = os.path.join(_script_dir, 'models', 'best.pt')
GENERAL_MODEL_PATH = os.path.join(_script_dir, 'models', 'yolov8n.pt')

# Use singleton model manager to avoid duplicate loading
deer_model = model_manager.get_deer_model(DEER_MODEL_PATH)
general_model = model_manager.get_general_model(GENERAL_MODEL_PATH)

# A set of common animal classes from the COCO dataset for easy lookup
ANIMAL_CLASSES = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}

# --- Utility Functions ---

# IoU calculation now imported from shared.math_utils

def get_detection_filename(timestamp=None):
    """Generate filename for detection video based on timestamp."""
    if timestamp is None:
        timestamp = datetime.now()
    date_str = timestamp.strftime('%Y-%m-%d')
    time_str = timestamp.strftime('%Y%m%d_%H%M%S')
    
    # Create date directory if it doesn't exist
    date_dir = os.path.join(DETECTIONS_FOLDER, date_str)
    os.makedirs(date_dir, exist_ok=True)
    
    video_filename = f"detection_{time_str}.mp4"
    metadata_filename = f"detection_{time_str}.json"
    
    return {
        'video_path': os.path.join(date_dir, video_filename),
        'metadata_path': os.path.join(date_dir, metadata_filename),
        'date_dir': date_dir
    }

def save_detection_metadata(metadata_path, detections, source_info, timestamp=None):
    """Save detection metadata to JSON file."""
    if timestamp is None:
        timestamp = datetime.now()
    
    metadata = {
        'timestamp': timestamp.isoformat(),
        'source': source_info,
        'detections': detections,
        'detection_count': len(detections),
        'has_deer': any(d['label'] == 'deer' for d in detections),
        'has_person': any(d['label'] == 'person' and d['confidence'] > 0.65 for d in detections),
        'config': {
            'confidence_threshold': MODEL_CONFIG['conf_threshold'],
            'model': 'best.pt'
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def should_save_detection(detections):
    """Determine if detections warrant saving a video clip."""
    if not DETECTION_VIDEO_CONFIG['save_detections']:
        return False
    
    # Check if any detection meets minimum confidence
    for det in detections:
        if det['confidence'] >= DETECTION_VIDEO_CONFIG['min_confidence']:
            # Save if deer detected or any animal when no person present
            if det['label'] == 'deer':
                return True
            # Check if it's an animal and no high-confidence person is present
            person_present = any(d['label'] == 'person' and d['confidence'] > 0.65 for d in detections)
            if det['label'] in ANIMAL_CLASSES and not person_present:
                return True
    
    return False

def cleanup_old_detections():
    """Remove detection videos older than configured retention period."""
    if not DETECTION_VIDEO_CONFIG['max_storage_days']:
        return
    
    from datetime import timedelta
    cutoff_date = datetime.now() - timedelta(days=DETECTION_VIDEO_CONFIG['max_storage_days'])
    removed_count = 0
    
    # Check each date directory
    for date_dir in os.listdir(DETECTIONS_FOLDER):
        date_path = os.path.join(DETECTIONS_FOLDER, date_dir)
        if not os.path.isdir(date_path):
            continue
            
        try:
            # Parse date from directory name
            dir_date = datetime.strptime(date_dir, '%Y-%m-%d')
            
            # Remove if older than cutoff
            if dir_date < cutoff_date:
                shutil.rmtree(date_path)
                removed_count += 1
                system_logger.info(f"Removed old detection directory: {date_dir}")
        except ValueError:
            # Skip directories that don't match date format
            continue
    
    if removed_count > 0:
        system_logger.info(f"Cleanup complete: removed {removed_count} old detection directories")

def filter_overlapping_detections(detections, iou_threshold=0.5):
    """
    Filter overlapping detections from the same model using NMS-style logic.
    Keeps the detection with higher confidence when boxes overlap significantly.
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    filtered_detections = []
    
    for current_det in sorted_detections:
        is_duplicate = False
        
        # Check if this detection overlaps significantly with any already accepted detection
        for accepted_det in filtered_detections:
            if calculate_iou(current_det['box'], accepted_det['box']) > iou_threshold:
                is_duplicate = True
                break
        
        # If not a duplicate, add to filtered list
        if not is_duplicate:
            filtered_detections.append(current_det)
    
    return filtered_detections

def get_available_cameras():
    """Checks for available camera devices."""
    indices = []
    camera_info = []
    
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera properties to help identify
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Try to read a frame to ensure camera is stable
            ret, _ = cap.read()
            
            if ret:
                indices.append(i)
                camera_info.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                camera_logger.info(f"Camera {i} detected: {width}x{height} @ {fps} FPS")
            else:
                camera_logger.warning(f"Camera {i} detected but couldn't read frame (possibly iPhone)")
                
            cap.release()
    
    # Log camera info for debugging
    camera_logger.info(f"Available cameras: {camera_info}")
    return indices

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.video = cv2.VideoCapture(self.camera_index)
        if not self.video.isOpened():
            raise RuntimeError("Could not start camera.")
        
        # Enable auto-adjustments for optimal image quality
        self._configure_auto_adjustments()

    def _configure_auto_adjustments(self):
        """Configure camera for automatic adjustments to lighting and focus."""
        camera_logger.info(f"Configuring auto-adjustments for camera {self.camera_index}")
        
        # Dictionary of properties to enable with their names for logging
        auto_properties = {
            'Auto Exposure': (cv2.CAP_PROP_AUTO_EXPOSURE, 3),  # 3 = auto mode
            'Auto Focus': (cv2.CAP_PROP_AUTOFOCUS, 1),        # 1 = enable
            'Auto White Balance': (cv2.CAP_PROP_AUTO_WB, 1),   # 1 = enable
            'Backlight Compensation': (cv2.CAP_PROP_BACKLIGHT, 1),  # 1 = enable
        }
        
        # Try to enable each property
        for prop_name, (prop_id, value) in auto_properties.items():
            try:
                success = self.video.set(prop_id, value)
                if success:
                    camera_logger.info(f"✓ {prop_name} enabled")
                else:
                    camera_logger.warning(f"✗ {prop_name} not supported or failed to enable")
            except Exception as e:
                camera_logger.warning(f"✗ {prop_name} error: {str(e)}")
        
        # Additional settings that might help with image quality
        try:
            # Set higher buffer size to reduce frame drops
            self.video.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            # Log current camera settings for debugging
            self._log_camera_properties()
        except Exception as e:
            camera_logger.debug(f"Additional settings error: {str(e)}")
    
    def _log_camera_properties(self):
        """Log current camera properties for debugging."""
        properties = {
            'Width': cv2.CAP_PROP_FRAME_WIDTH,
            'Height': cv2.CAP_PROP_FRAME_HEIGHT,
            'FPS': cv2.CAP_PROP_FPS,
            'Brightness': cv2.CAP_PROP_BRIGHTNESS,
            'Contrast': cv2.CAP_PROP_CONTRAST,
            'Saturation': cv2.CAP_PROP_SATURATION,
            'Exposure': cv2.CAP_PROP_EXPOSURE,
            'Gain': cv2.CAP_PROP_GAIN,
        }
        
        camera_logger.debug(f"Camera {self.camera_index} current settings:")
        for prop_name, prop_id in properties.items():
            try:
                value = self.video.get(prop_id)
                if value != -1:  # -1 typically means not supported
                    camera_logger.debug(f"  {prop_name}: {value}")
            except:
                pass

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        return frame if success else None
    
    def get_properties(self):
        """Get current camera properties as a dictionary."""
        props = {}
        property_map = {
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS,
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'gain': cv2.CAP_PROP_GAIN,
            'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
            'auto_wb': cv2.CAP_PROP_AUTO_WB,
            'autofocus': cv2.CAP_PROP_AUTOFOCUS,
        }
        
        for name, prop_id in property_map.items():
            try:
                value = self.video.get(prop_id)
                if value != -1:
                    props[name] = value
            except:
                pass
        
        return props

# --- Drawing Helpers ---

def _draw_bounding_box(frame, box, label, conf, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_text = f"{label} {conf:.2f}"
    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def _draw_bullseye(frame, target):
    box = target['box']
    center_x = int((box[0] + box[2]) / 2)
    center_y = int((box[1] + box[3]) / 2)
    cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), 2)
    cv2.circle(frame, (center_x, center_y), 20, (0, 0, 255), 2)
    cv2.line(frame, (center_x - 25, center_y), (center_x + 25, center_y), (0, 0, 255), 2)
    cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 25), (0, 0, 255), 2)

# --- Core Inference Logic ---

def run_inference_on_frame(frame, use_cascaded=None, conf_threshold=None):
    """
    Run object detection on a frame with optional cascaded inference.
    
    Args:
        frame: Input image frame
        use_cascaded: If True, only run deer model when animals are detected
        conf_threshold: Confidence threshold for detections
    """
    # Use config values if not specified
    if use_cascaded is None:
        use_cascaded = MODEL_CONFIG['use_cascaded']
    if conf_threshold is None:
        conf_threshold = MODEL_CONFIG['conf_threshold']
    
    # Always run general model first with optimization settings
    general_results = general_model(
        frame, 
        verbose=False,
        imgsz=MODEL_CONFIG['imgsz'],
        device=MODEL_CONFIG['device'],
        half=MODEL_CONFIG['half'],
        max_det=MODEL_CONFIG['max_det'],
        conf=conf_threshold,
        iou=MODEL_CONFIG['iou_threshold']
    )
    general_detections = []
    for r in general_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                # Don't double-check confidence, model already filtered
                general_detections.append({
                    'box': b.xyxy[0].tolist(),
                    'conf': float(b.conf[0]),
                    'cls': int(b.cls[0]),
                    'label': general_model.names[int(b.cls[0])]
                })
    
    # Filter overlapping detections within the general model (less aggressive)
    general_detections = filter_overlapping_detections(general_detections, iou_threshold=0.8)
    
    # Check if we should run deer model (cascaded approach)
    deer_detections = []
    if use_cascaded:
        # Only run deer model if animals are detected by general model
        animal_detected = any(d['label'] in ANIMAL_CLASSES or d['label'] == 'deer' for d in general_detections)
        if animal_detected:
            deer_results = deer_model(
                frame, 
                verbose=False,
                imgsz=MODEL_CONFIG['imgsz'],
                device=MODEL_CONFIG['device'],
                half=MODEL_CONFIG['half'],
                max_det=MODEL_CONFIG['max_det'],
                conf=conf_threshold,
                iou=MODEL_CONFIG['iou_threshold']
            )
            deer_detections = []
            for r in deer_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for b in r.boxes:
                        # Don't double-check confidence, model already filtered
                        deer_detections.append({
                            'box': b.xyxy[0].tolist(),
                            'conf': float(b.conf[0]),
                            'cls': int(b.cls[0]),
                            'label': deer_model.names[int(b.cls[0])]
                        })
            # Filter overlapping detections within the deer model first (less aggressive - only filter true duplicates)
            deer_detections = filter_overlapping_detections(deer_detections, iou_threshold=0.8)
    else:
        # Original behavior - always run both models
        deer_results = deer_model(
            frame, 
            verbose=False,
            imgsz=MODEL_CONFIG['imgsz'],
            device=MODEL_CONFIG['device'],
            half=MODEL_CONFIG['half'],
            max_det=MODEL_CONFIG['max_det'],
            conf=conf_threshold,
            iou=MODEL_CONFIG['iou_threshold']
        )
        deer_detections = []
        for r in deer_results:
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    # Don't double-check confidence, model already filtered
                    deer_detections.append({
                        'box': b.xyxy[0].tolist(),
                        'conf': float(b.conf[0]),
                        'cls': int(b.cls[0]),
                        'label': deer_model.names[int(b.cls[0])]
                    })
        # Filter overlapping detections within the deer model first (less aggressive - only filter true duplicates)
        deer_detections = filter_overlapping_detections(deer_detections, iou_threshold=0.8)

    # Start with filtered deer detections
    final_detections = list(deer_detections)
    
    # Add non-overlapping general detections (but always keep people, deer model always wins)
    for gen_det in general_detections:
        is_overlapping = False
        if gen_det['label'] != 'person':
            for deer_det in deer_detections:
                # More aggressive filtering - deer model wins with lower IoU threshold
                if calculate_iou(gen_det['box'], deer_det['box']) > 0.3:
                    is_overlapping = True
                    break
        if not is_overlapping:
            final_detections.append(gen_det)

    person_present = any(d['label'] == 'person' and d['conf'] > 0.65 for d in final_detections)
    largest_target = None
    if not person_present:
        largest_area = 0
        for det in final_detections:
            if det['label'] == 'deer' or det['label'] in ANIMAL_CLASSES:
                box = det['box']
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > largest_area:
                    largest_area = area
                    largest_target = det

    annotated_frame = frame.copy()
    
    # Debug: print detection count
    if final_detections:
        print(f"DEBUG: Drawing {len(final_detections)} detections")
    
    for det in final_detections:
        label, color = det['label'], (255, 0, 0)
        if person_present:
            if det['label'] == 'person':
                color = (0, 255, 0)
            elif det['label'] == 'deer' or det['label'] in ANIMAL_CLASSES:
                label = 'Pet'
                color = (255, 0, 0)
        else:
            if det['label'] == 'deer':
                color = (0, 0, 255)
            elif det['label'] in ANIMAL_CLASSES:
                color = (255, 0, 0)
        
        # Debug: print what we're drawing
        print(f"DEBUG: Drawing {label} at box {det['box'][:2]}...")
        _draw_bounding_box(annotated_frame, det['box'], label, det['conf'], color)

    if largest_target:
        _draw_bullseye(annotated_frame, largest_target)

    # For video/live feed, return full detection info; for display, return summary
    detection_summary = [{'label': d['label'], 'confidence': d['conf']} for d in final_detections]
    detection_summary.sort(key=lambda x: x['confidence'], reverse=True)
    return annotated_frame, detection_summary

def run_inference(image_path):
    img = cv2.imread(image_path)
    return run_inference_on_frame(img) if img is not None else (None, [])

def run_inference_video(video_path, output_path, save_detections=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, []
    
    try:
        width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
        
        # Simple ffmpeg command
        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', 
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps), 
            '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_path
        ]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        summary = []
        frame_count = 0
        detection_found = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Run inference on every frame (simple approach)
            annotated_frame, detections = run_inference_on_frame(frame)
            
            # Check if we should save this as a detection
            if save_detections and detections and should_save_detection(detections):
                detection_found = True
            
            # Debug output for first frame
            if frame_count == 0 and detections:
                print(f"DEBUG: First frame detections count: {len(detections)}")
                print(f"DEBUG: First detection: {detections[0] if detections else 'None'}")
            
            if annotated_frame is None: 
                annotated_frame = frame
                
            summary.extend(detections)
            frame_count += 1
            
            try:
                proc.stdin.write(annotated_frame.tobytes())
            except (IOError, BrokenPipeError):
                print(f"ffmpeg error: {proc.stderr.read().decode()}")
                break
        
    finally:
        # Ensure proper cleanup
        cap.release()
        if 'proc' in locals() and proc.stdin:
            proc.stdin.close()
            proc.wait()
    
    # If detection found, save a copy to detections folder
    if save_detections and detection_found and os.path.exists(output_path):
        timestamp = datetime.now()
        detection_files = get_detection_filename(timestamp)
        
        # Copy the processed video to detections folder
        shutil.copy2(output_path, detection_files['video_path'])
        
        # Save metadata
        source_info = {
            'type': 'uploaded_video',
            'original_filename': os.path.basename(video_path),
            'processed_path': output_path
        }
        
        unique_detections = list({(d['label'], d['confidence']) for d in summary})
        detection_summary = [{'label': l, 'confidence': c} for l, c in unique_detections]
        save_detection_metadata(detection_files['metadata_path'], detection_summary, source_info, timestamp)
        
        # Log detection event
        logger_config.log_detection_event(detection_summary, source_info['type'], detection_files['video_path'])
        app_logger.info(f"Detection video saved: {detection_files['video_path']}")
        
    unique_summary = sorted(list({(d['label'], f"{d['confidence']:.2f}") for d in summary}), key=lambda x: float(x[1]), reverse=True)
    return output_path, [{'label': l, 'confidence': c} for l, c in unique_summary]

# --- Flask Routes ---

@app.route('/cameras')
def list_cameras():
    return jsonify(get_available_cameras())

@app.route('/camera_test')
def camera_test():
    return render_template('camera_test.html')

@app.route('/')
def index():
    return render_template('index.html')

def gen_video_feed(camera):
    frame_count = 0
    frame_skip = MODEL_CONFIG['frame_skip']
    last_annotated_frame = None
    
    # For detection video saving
    detection_buffer = []
    buffer_fps = 10  # Assumed FPS for live camera
    max_buffer_frames = int(buffer_fps * (DETECTION_VIDEO_CONFIG['pre_detection_seconds'] + 
                                          DETECTION_VIDEO_CONFIG['clip_duration_seconds']))
    detection_active = False
    detection_end_frame = 0
    post_detection_frames = int(buffer_fps * DETECTION_VIDEO_CONFIG['post_detection_seconds'])
    
    while True:
        frame = camera.get_frame()
        if frame is None: break
        
        # Process frame based on frame skip setting
        if frame_count % frame_skip == 0:
            # Run inference on this frame
            annotated_frame, detections = run_inference_on_frame(frame)
            last_annotated_frame = annotated_frame
            
            # Check for detections to save
            if DETECTION_VIDEO_CONFIG['save_detections'] and detections and should_save_detection(detections):
                if not detection_active:
                    detection_active = True
                    print(f"Detection started at frame {frame_count}")
                detection_end_frame = frame_count + post_detection_frames
        else:
            # Use last annotated frame if available, otherwise use current frame
            annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame
            detections = []  # No new detections on skipped frames
        
        # Add frame to buffer for potential detection saving
        if DETECTION_VIDEO_CONFIG['save_detections']:
            detection_buffer.append((annotated_frame.copy(), detections))
            
            # Keep buffer size limited
            if len(detection_buffer) > max_buffer_frames:
                detection_buffer.pop(0)
            
            # Save detection video when detection period ends
            if detection_active and frame_count >= detection_end_frame:
                save_camera_detection_clip(detection_buffer, camera.camera_index)
                detection_active = False
                detection_buffer = []  # Clear buffer after saving
        
        frame_count += 1
        
        (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

def save_camera_detection_clip(buffer, camera_index):
    """Save detection clip from camera buffer."""
    if not buffer:
        return
    
    timestamp = datetime.now()
    detection_files = get_detection_filename(timestamp)
    
    # Get frame dimensions
    height, width = buffer[0][0].shape[:2]
    fps = 10  # Assumed FPS
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(detection_files['video_path'], fourcc, fps, (width, height))
    
    all_detections = []
    for frame, detections in buffer:
        out.write(frame)
        if detections:
            all_detections.extend(detections)
    
    out.release()
    
    # Save metadata
    source_info = {
        'type': 'live_camera',
        'camera_index': camera_index,
        'duration_frames': len(buffer),
        'fps': fps
    }
    
    unique_detections = list({(d['label'], d['confidence']) for d in all_detections})
    detection_summary = [{'label': l, 'confidence': c} for l, c in unique_detections]
    save_detection_metadata(detection_files['metadata_path'], detection_summary, source_info, timestamp)
    
    # Log detection event
    logger_config.log_detection_event(detection_summary, source_info['type'], detection_files['video_path'])
    camera_logger.info(f"Camera detection video saved: {detection_files['video_path']}")

@app.route('/video_feed')
def video_feed():
    camera_index = request.args.get('camera_index', 0, type=int)
    return Response(gen_video_feed(Camera(camera_index=camera_index)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file or not file.filename:
        return 'No selected file'
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        annotated_image, detections = run_inference(filepath)
        if annotated_image is None:
            return "Error processing image.", 500
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return render_template('result.html', result_path=img_str, detections=detections, is_image=True)
    
    elif filename.lower().endswith(('.mp4', '.avi', 'mov')):
        output_filename = 'processed_' + filename
        output_path = os.path.join(STATIC_FOLDER, output_filename)
        processed_path, detections = run_inference_video(filepath, output_path)
        if processed_path:
            video_url = url_for('static', filename=output_filename)
            return render_template('result_video.html', video_path=video_url, detections=detections)
        else:
            return "Error processing video.", 500

    return 'Invalid file type'

@app.route('/detections')
def list_detections():
    """List all saved detection videos."""
    detections = []
    
    # Walk through detection directories
    for date_dir in sorted(os.listdir(DETECTIONS_FOLDER), reverse=True):
        date_path = os.path.join(DETECTIONS_FOLDER, date_dir)
        if not os.path.isdir(date_path):
            continue
            
        # Get all detection files for this date
        for filename in sorted(os.listdir(date_path), reverse=True):
            if filename.endswith('.json'):
                metadata_path = os.path.join(date_path, filename)
                video_filename = filename.replace('.json', '.mp4')
                video_path = os.path.join(date_path, video_filename)
                
                if os.path.exists(video_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    detections.append({
                        'date': date_dir,
                        'filename': video_filename,
                        'timestamp': metadata['timestamp'],
                        'has_deer': metadata.get('has_deer', False),
                        'has_person': metadata.get('has_person', False),
                        'detection_count': metadata.get('detection_count', 0),
                        'source': metadata.get('source', {}).get('type', 'unknown')
                    })
    
    return render_template('detections.html', detections=detections)

@app.route('/detections/<date>/<filename>')
def view_detection(date, filename):
    """View a specific detection video."""
    video_path = os.path.join(DETECTIONS_FOLDER, date, filename)
    metadata_path = video_path.replace('.mp4', '.json')
    
    if not os.path.exists(video_path):
        return "Detection not found", 404
    
    # Load metadata
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Create a URL for the video
    # For now, we'll copy it to static folder for serving
    static_filename = f"detection_{date}_{filename}"
    static_path = os.path.join(STATIC_FOLDER, static_filename)
    shutil.copy2(video_path, static_path)
    
    video_url = url_for('static', filename=static_filename)
    
    return render_template('view_detection.html', 
                         video_path=video_url, 
                         metadata=metadata,
                         date=date,
                         filename=filename)

# --- Monitoring & Debug Routes ---

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    try:
        status = {
            'system': system_monitor.get_system_info(),
            'process': system_monitor.get_process_info(),
            'disk_space': system_monitor.check_disk_space(),
            'detection_stats': system_monitor.get_detection_stats(DETECTIONS_FOLDER),
            'cameras': get_available_cameras(),
            'debug_mode': logger_config.log_level == logging.DEBUG
        }
        return jsonify(status)
    except Exception as e:
        app_logger.error(f"Error getting system status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/<log_type>')
def api_logs(log_type):
    """API endpoint for retrieving logs."""
    lines = request.args.get('lines', 100, type=int)
    logs = logger_config.get_recent_logs(log_type, lines)
    return jsonify({
        'log_type': log_type,
        'lines': len(logs),
        'content': logs
    })

@app.route('/status')
def status_page():
    """Web page for system status monitoring."""
    return render_template('status.html')

@app.route('/logs')
def logs_page():
    """Web page for viewing logs."""
    return render_template('logs.html')

@app.route('/api/debug', methods=['POST'])
def toggle_debug():
    """Toggle debug mode."""
    data = request.get_json()
    enabled = data.get('enabled', False)
    logger_config.set_debug_mode(enabled)
    app_logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
    return jsonify({'debug_mode': enabled})

@app.route('/api/camera/<int:camera_index>/properties')
def get_camera_properties(camera_index):
    """Get properties for a specific camera."""
    try:
        camera = Camera(camera_index)
        properties = camera.get_properties()
        del camera  # Release the camera
        return jsonify({
            'camera_index': camera_index,
            'properties': properties,
            'auto_adjustments': {
                'auto_exposure': properties.get('auto_exposure', -1) == 3,
                'auto_focus': properties.get('autofocus', -1) == 1,
                'auto_white_balance': properties.get('auto_wb', -1) == 1,
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Main Execution ---
if __name__ == '__main__':
    # Log startup
    app_logger.info("="*50)
    app_logger.info("Smart Deer Deterrent System Starting")
    app_logger.info(f"Version: 1.0.0")
    app_logger.info(f"Detection saving: {DETECTION_VIDEO_CONFIG['save_detections']}")
    app_logger.info(f"Models: {os.path.basename(DEER_MODEL_PATH)}, {os.path.basename(GENERAL_MODEL_PATH)}")
    app_logger.info("="*50)
    
    # Run cleanup on startup
    try:
        cleanup_old_detections()
    except Exception as e:
        app_logger.error(f"Error during cleanup: {str(e)}")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5001, debug=True) 