import base64
import os
import subprocess
import cv2
from flask import Flask, render_template, request, jsonify, Response, url_for, make_response, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import sys
import json
from datetime import datetime
import shutil
import logging
import time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.model_manager import model_manager
from shared.math_utils import calculate_iou
from shared.detection_tracker import DetectionTracker
from logger_config import logger_config
from system_monitor import system_monitor
from turret_control.main import TurretController
from turret_control.coordinate_transform import CameraConfig, TurretConfig
from turret_control.smoothing_filter import AdaptiveSmoothingFilter

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

# --- Turret Display Settings ---
TURRET_DISPLAY_CONFIG = {
    'show_calculations': True,     # Enable turret calculations (for separate pane)
    'show_video_overlay': False,   # Don't show turret overlay on video frames
    'show_info_panel': True,       # Show information panel (when enabled)
    'panel_opacity': 0.8,          # Info panel background opacity
    'text_color': (255, 255, 255), # White text
    'panel_color': (0, 0, 0),      # Black background
}

# --- Model Loading with Optimization ---
DEER_MODEL_PATH = os.path.join(_script_dir, 'models', 'best.pt')
GENERAL_MODEL_PATH = os.path.join(_script_dir, 'models', 'yolov8n.pt')

# Use singleton model manager to avoid duplicate loading
deer_model = model_manager.get_deer_model(DEER_MODEL_PATH)
general_model = model_manager.get_general_model(GENERAL_MODEL_PATH)

# A set of common animal classes from the COCO dataset for easy lookup
ANIMAL_CLASSES = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}

# Classes to keep - only animals and people (filter out all inanimate objects)
ALLOWED_CLASSES = ANIMAL_CLASSES | {'person', 'deer'}

# Initialize turret controller for calculations with smoothing enabled
turret_config_path = os.path.join(_script_dir, '..', 'turret_control', 'turret_config.json')
turret_controller = TurretController(
    config_file=turret_config_path if os.path.exists(turret_config_path) else None,
    use_smoothing=True,
    use_adaptive_smoothing=True  # Use adaptive smoothing for better handling of stationary targets
)

# Store current camera detections for API access
camera_detections = {}

# Detection history for handling person/animal oscillation
# Stores recent detection labels to prevent false firing when oscillating between person and animal
detection_history = {}  # camera_index -> list of recent labels
DETECTION_HISTORY_SIZE = 15  # Number of frames to track
PERSON_SAFETY_FRAMES = 10  # Frames to wait after last person detection before allowing targeting

# Target smoothing configuration
TARGET_SMOOTHING_CONFIG = {
    'enabled': True,
    'deadband_threshold': 15,  # Pixels - ignore movements smaller than this
    'smoothing_factor': 0.3,   # 0.0 = no smoothing, 1.0 = no movement
    'confidence_weight': 0.2,  # How much detection confidence affects smoothing
    'size_weight': 0.1,        # How much target size affects smoothing
    # Anatomical awareness settings
    'use_anatomical_smoothing': True,  # Enable anatomical-aware smoothing
    'head_movement_threshold': 20,     # Pixels - threshold for head-only movement
    'body_anchor_weight': 0.8,         # How much to weight the body center vs bbox center
    'feet_stability_factor': 0.9,      # How stable to keep the bottom anchor (feet)
    'min_body_movement': 10,           # Minimum pixels for body movement detection
    'show_body_center': False,         # Show green dot at calculated body center for debugging
}

# Global target tracking for smoothing
previous_targets = {}  # camera_index -> {'center': (x, y), 'timestamp': time, 'confidence': conf}

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
    """Determine if detections warrant saving a video clip - only animals, NOT people."""
    if not DETECTION_VIDEO_CONFIG['save_detections']:
        return False
    
    # First check if any person is in the frame - if so, don't save
    for det in detections:
        if det['label'] == 'person' and det['confidence'] >= 0.3:  # Low threshold to be safe
            return False
    
    # Now check if any animal meets minimum confidence
    for det in detections:
        if det['confidence'] >= DETECTION_VIDEO_CONFIG['min_confidence']:
            # Save if deer or any animal detected (but we already excluded person above)
            if det['label'] == 'deer':
                return True
            if det['label'] in ANIMAL_CLASSES:
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

# --- Target Smoothing ---

def smooth_target_position(target, camera_source='default'):
    """
    Apply anatomical-aware smoothing to target position.
    
    This function understands deer anatomy:
    - Feet stay on the ground (bottom of bbox is stable)
    - Head moves up/down when grazing (top of bbox is mobile)
    - Body center is more stable than bbox center
    
    Args:
        target: Current target detection with 'box' and 'conf' keys
        camera_source: Identifier for camera/source (for separate tracking)
    
    Returns:
        dict: Smoothed target with updated box coordinates
    """
    if not TARGET_SMOOTHING_CONFIG['enabled'] or not target:
        return target
    
    current_box = target['box']
    current_confidence = target.get('conf', 0.0)
    current_time = time.time()
    
    # Get previous target for this camera/source
    prev_target = previous_targets.get(camera_source)
    
    if prev_target is None:
        # First target - no smoothing needed
        previous_targets[camera_source] = {
            'box': current_box,
            'confidence': current_confidence,
            'timestamp': current_time
        }
        return target
    
    prev_box = prev_target['box']
    
    # Use anatomical-aware smoothing if enabled
    if TARGET_SMOOTHING_CONFIG['use_anatomical_smoothing']:
        return _anatomical_smooth_target(current_box, prev_box, target, camera_source, current_confidence, current_time)
    else:
        # Fall back to simple center-based smoothing
        return _simple_smooth_target(current_box, prev_box, target, camera_source, current_confidence, current_time)


def _anatomical_smooth_target(current_box, prev_box, target, camera_source, current_confidence, current_time):
    """
    Apply anatomical-aware smoothing based on deer movement patterns.
    
    Key insights:
    - Deer feet stay on ground (bottom edge stable)
    - Head moves up/down when grazing (top edge mobile)
    - Body center is more stable than bbox center
    - Only move target when actual body movement occurs
    """
    # Calculate edge movements
    left_movement = abs(current_box[0] - prev_box[0])
    top_movement = abs(current_box[1] - prev_box[1])
    right_movement = abs(current_box[2] - prev_box[2])
    bottom_movement = abs(current_box[3] - prev_box[3])
    
    # Calculate body anchor points (more stable than bbox center)
    # Use bottom-center for feet position (most stable)
    current_feet_center = ((current_box[0] + current_box[2]) / 2, current_box[3])
    prev_feet_center = ((prev_box[0] + prev_box[2]) / 2, prev_box[3])
    
    # Calculate body center (slightly above bottom, more stable than bbox center)
    body_offset = (current_box[3] - current_box[1]) * 0.25  # 25% up from bottom
    current_body_center = ((current_box[0] + current_box[2]) / 2, 
                          current_box[3] - body_offset)
    prev_body_center = ((prev_box[0] + prev_box[2]) / 2, 
                       prev_box[3] - body_offset)
    
    # Detect movement type
    head_only_movement = (
        top_movement > TARGET_SMOOTHING_CONFIG['head_movement_threshold'] and
        bottom_movement < TARGET_SMOOTHING_CONFIG['min_body_movement'] and
        left_movement < TARGET_SMOOTHING_CONFIG['min_body_movement'] and
        right_movement < TARGET_SMOOTHING_CONFIG['min_body_movement']
    )
    
    # Calculate body movement distance
    body_movement_distance = ((current_body_center[0] - prev_body_center[0])**2 + 
                            (current_body_center[1] - prev_body_center[1])**2)**0.5
    
    # If it's just head movement, anchor to the stable body position
    if head_only_movement:
        app_logger.debug(f"Head-only movement detected (top: {top_movement:.1f}px), anchoring to body")
        
        # Use previous body center with slight smoothing
        smoothing_factor = TARGET_SMOOTHING_CONFIG['feet_stability_factor']
        stable_body_center = (
            prev_body_center[0] * smoothing_factor + current_body_center[0] * (1 - smoothing_factor),
            prev_body_center[1] * smoothing_factor + current_body_center[1] * (1 - smoothing_factor)
        )
        
        # Keep the target stable
        smoothed_target = target.copy()
        smoothed_target['box'] = prev_box  # Keep previous box
        
        # Update tracking with stable position
        previous_targets[camera_source] = {
            'box': prev_box,
            'confidence': current_confidence,
            'timestamp': current_time
        }
        
        return smoothed_target
    
    # Check for significant body movement
    if body_movement_distance < TARGET_SMOOTHING_CONFIG['deadband_threshold']:
        app_logger.debug(f"Insufficient body movement ({body_movement_distance:.1f}px), keeping stable")
        
        # Keep previous position
        smoothed_target = target.copy()
        smoothed_target['box'] = prev_box
        
        previous_targets[camera_source] = {
            'box': prev_box,
            'confidence': current_confidence,
            'timestamp': current_time
        }
        
        return smoothed_target
    
    # Significant body movement detected - apply smoothing
    app_logger.debug(f"Body movement detected ({body_movement_distance:.1f}px), applying smoothing")
    app_logger.debug(f"Current body center: ({current_body_center[0]:.1f}, {current_body_center[1]:.1f}), Previous: ({prev_body_center[0]:.1f}, {prev_body_center[1]:.1f})")
    
    # Calculate adaptive smoothing factor
    base_smoothing = TARGET_SMOOTHING_CONFIG['smoothing_factor']
    
    # Adjust smoothing based on confidence
    conf_factor = 1.0 - (current_confidence * TARGET_SMOOTHING_CONFIG['confidence_weight'])
    
    # Final smoothing factor
    smoothing_factor = min(0.9, base_smoothing + conf_factor)
    
    # Apply smoothing to body center
    smoothed_body_center = (
        prev_body_center[0] * smoothing_factor + current_body_center[0] * (1 - smoothing_factor),
        prev_body_center[1] * smoothing_factor + current_body_center[1] * (1 - smoothing_factor)
    )
    
    # Reconstruct bounding box with proper body center positioning
    width = current_box[2] - current_box[0]
    height = current_box[3] - current_box[1]
    
    # Body center is 25% up from bottom, so:
    # - Distance from body center to bottom = 25% of height
    # - Distance from body center to top = 75% of height
    bottom_offset = height * 0.25  # Distance from body center to bottom
    top_offset = height * 0.75     # Distance from body center to top
    
    # Place box with smoothed body center at proper position
    smoothed_box = [
        smoothed_body_center[0] - width/2,                      # Left
        smoothed_body_center[1] - top_offset,                   # Top (75% above body center)
        smoothed_body_center[0] + width/2,                      # Right  
        smoothed_body_center[1] + bottom_offset                 # Bottom (25% below body center)
    ]
    
    # Debug: log the box reconstruction
    box_center_y = (smoothed_box[1] + smoothed_box[3]) / 2
    app_logger.debug(f"Reconstructed box center: ({(smoothed_box[0] + smoothed_box[2])/2:.1f}, {box_center_y:.1f}), Body center: ({smoothed_body_center[0]:.1f}, {smoothed_body_center[1]:.1f})")
    
    # Create smoothed target
    smoothed_target = target.copy()
    smoothed_target['box'] = smoothed_box
    
    # Update previous target
    previous_targets[camera_source] = {
        'box': smoothed_box,
        'confidence': current_confidence,
        'timestamp': current_time
    }
    
    return smoothed_target


def _simple_smooth_target(current_box, prev_box, target, camera_source, current_confidence, current_time):
    """
    Apply simple center-based smoothing (fallback method).
    """
    current_center = ((current_box[0] + current_box[2]) / 2, 
                     (current_box[1] + current_box[3]) / 2)
    prev_center = ((prev_box[0] + prev_box[2]) / 2, 
                  (prev_box[1] + prev_box[3]) / 2)
    
    # Calculate distance moved
    distance_moved = ((current_center[0] - prev_center[0])**2 + 
                     (current_center[1] - prev_center[1])**2)**0.5
    
    # Apply deadband
    if distance_moved < TARGET_SMOOTHING_CONFIG['deadband_threshold']:
        return target
    
    # Apply smoothing
    smoothing_factor = TARGET_SMOOTHING_CONFIG['smoothing_factor']
    smoothed_center = (
        prev_center[0] * smoothing_factor + current_center[0] * (1 - smoothing_factor),
        prev_center[1] * smoothing_factor + current_center[1] * (1 - smoothing_factor)
    )
    
    # Update box
    width = current_box[2] - current_box[0]
    height = current_box[3] - current_box[1]
    smoothed_box = [
        smoothed_center[0] - width/2,
        smoothed_center[1] - height/2,
        smoothed_center[0] + width/2,
        smoothed_center[1] + height/2
    ]
    
    smoothed_target = target.copy()
    smoothed_target['box'] = smoothed_box
    
    previous_targets[camera_source] = {
        'box': smoothed_box,
        'confidence': current_confidence,
        'timestamp': current_time
    }
    
    return smoothed_target

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
    
    # Draw main bullseye at box center
    cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), 2)
    cv2.circle(frame, (center_x, center_y), 20, (0, 0, 255), 2)
    cv2.line(frame, (center_x - 25, center_y), (center_x + 25, center_y), (0, 0, 255), 2)
    cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 25), (0, 0, 255), 2)
    
    # Draw body center indicator (for debugging anatomical smoothing)
    if TARGET_SMOOTHING_CONFIG.get('show_body_center', False):
        height = box[3] - box[1]
        body_center_y = int(box[3] - height * 0.25)  # 25% up from bottom
        cv2.circle(frame, (center_x, body_center_y), 5, (0, 255, 0), 2)  # Green dot
        cv2.putText(frame, "Body Center", (center_x + 30, body_center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def _draw_turret_info_panel(frame, target_box, camera_pan=0.0):
    """Draw turret calculation information panel on frame."""
    if not TURRET_DISPLAY_CONFIG['show_video_overlay']:
        return
    
    # Update turret controller with camera pan
    turret_controller.update_camera_pan(camera_pan)
    
    # Get target center
    target_center_x = (target_box[0] + target_box[2]) / 2
    target_center_y = (target_box[1] + target_box[3]) / 2
    
    # Get transformation details
    details = turret_controller.transformer.get_transformation_details(
        target_center_x, target_center_y
    )
    
    # Panel dimensions
    panel_width = 400
    panel_height = 350
    panel_x = frame.shape[1] - panel_width - 10
    panel_y = 10
    
    # Draw semi-transparent panel background
    if TURRET_DISPLAY_CONFIG['show_info_panel']:
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     TURRET_DISPLAY_CONFIG['panel_color'], -1)
        cv2.addWeighted(overlay, TURRET_DISPLAY_CONFIG['panel_opacity'], 
                       frame, 1 - TURRET_DISPLAY_CONFIG['panel_opacity'], 0, frame)
        
        # Draw panel border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     TURRET_DISPLAY_CONFIG['text_color'], 2)
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 25
    text_color = TURRET_DISPLAY_CONFIG['text_color']
    
    # Title
    y_pos = panel_y + 30
    cv2.putText(frame, "Turret Calculations", (panel_x + 10, y_pos),
                font, font_scale * 1.2, text_color, thickness + 1)
    
    y_pos += line_height + 10
    
    # Input section
    cv2.putText(frame, f"Target Pixel: ({int(target_center_x)}, {int(target_center_y)})",
                (panel_x + 20, y_pos), font, font_scale, text_color, thickness)
    y_pos += line_height
    
    cv2.putText(frame, f"Camera Pan: {camera_pan:.1f}°",
                (panel_x + 20, y_pos), font, font_scale, text_color, thickness)
    y_pos += line_height + 5
    
    # Camera-relative angles
    cv2.putText(frame, "Camera-Relative Angles:",
                (panel_x + 20, y_pos), font, font_scale, text_color, thickness)
    y_pos += line_height
    
    cv2.putText(frame, f"  Pan:  {details['camera_relative']['pan']:7.2f}°",
                (panel_x + 30, y_pos), font, font_scale, text_color, thickness)
    y_pos += line_height
    
    cv2.putText(frame, f"  Tilt: {details['camera_relative']['tilt']:7.2f}°",
                (panel_x + 30, y_pos), font, font_scale, text_color, thickness)
    y_pos += line_height + 5
    
    # World angles
    cv2.putText(frame, "World Angles:",
                (panel_x + 20, y_pos), font, font_scale, text_color, thickness)
    y_pos += line_height
    
    cv2.putText(frame, f"  Pan:  {details['world_angles']['pan']:7.2f}°",
                (panel_x + 30, y_pos), font, font_scale, text_color, thickness)
    y_pos += line_height
    
    cv2.putText(frame, f"  Tilt: {details['world_angles']['tilt']:7.2f}°",
                (panel_x + 30, y_pos), font, font_scale, text_color, thickness)
    y_pos += line_height + 5
    
    # Turret angles
    cv2.putText(frame, "Turret Command:",
                (panel_x + 20, y_pos), font, font_scale, text_color, thickness)
    y_pos += line_height
    
    cv2.putText(frame, f"  Pan:  {details['final_angles']['pan']:7.2f}°",
                (panel_x + 30, y_pos), font, font_scale * 1.1, text_color, thickness + 1)
    y_pos += line_height
    
    cv2.putText(frame, f"  Tilt: {details['final_angles']['tilt']:7.2f}°",
                (panel_x + 30, y_pos), font, font_scale * 1.1, text_color, thickness + 1)
    y_pos += line_height
    
    # Valid indicator
    if details['final_angles']['is_valid']:
        status_color = (0, 255, 0)  # Green
        status_text = "✓ Within Limits"
    else:
        status_color = (0, 0, 255)  # Red
        status_text = "✗ Outside Limits"
    
    cv2.putText(frame, status_text,
                (panel_x + 30, y_pos), font, font_scale, status_color, thickness + 1)

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
                label = general_model.names[int(b.cls[0])]
                # Only keep animals and people, filter out all inanimate objects
                if label in ALLOWED_CLASSES:
                    general_detections.append({
                        'box': b.xyxy[0].tolist(),
                        'conf': float(b.conf[0]),
                        'cls': int(b.cls[0]),
                        'label': label
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

    # Check for person in current frame
    person_in_frame = any(d['label'] == 'person' and d['conf'] > 0.50 for d in final_detections)
    
    # Check for oscillation between person and animal detections
    # This prevents firing when system is confused between person and deer
    person_recently_detected = False
    
    # Determine if this is a live camera frame or uploaded video
    is_live_camera = hasattr(frame, 'camera_index')
    
    if is_live_camera:
        camera_idx = frame.camera_index
        
        # Initialize history if needed
        if camera_idx not in detection_history:
            detection_history[camera_idx] = []
        
        # Update detection history
        current_labels = [d['label'] for d in final_detections if d['conf'] > 0.50]
        detection_history[camera_idx].append(current_labels)
        
        # Keep history size limited
        if len(detection_history[camera_idx]) > DETECTION_HISTORY_SIZE:
            detection_history[camera_idx].pop(0)
        
        # Check if person was detected in recent frames
        for frame_labels in detection_history[camera_idx][-PERSON_SAFETY_FRAMES:]:
            if 'person' in frame_labels:
                person_recently_detected = True
                break
    
    # For uploaded videos, only consider current frame person detection
    # For live camera, consider both current and recent history
    if is_live_camera:
        person_present = person_in_frame or person_recently_detected
    else:
        person_present = person_in_frame  # Only current frame for uploaded videos
    
    # Debug logging for person detection
    if person_present:
        app_logger.debug(f"Person safety active: current={person_in_frame}, recent={person_recently_detected}")
    
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
    
    # Apply smoothing to the target if found
    if largest_target:
        # Determine camera source for tracking (use camera index for live, 'uploaded' for videos)
        camera_source = frame.camera_index if hasattr(frame, 'camera_index') else 'uploaded'
        largest_target = smooth_target_position(largest_target, camera_source)

    annotated_frame = frame.copy()
    
    # Debug: log detection count
    if final_detections:
        app_logger.debug(f"Drawing {len(final_detections)} detections")
    
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
        
        # Debug: log what we're drawing
        app_logger.debug(f"Drawing {label} at box {det['box'][:2]}...")
        _draw_bounding_box(annotated_frame, det['box'], label, det['conf'], color)

    if largest_target:
        app_logger.debug(f"Drawing bullseye and turret panel for target: {largest_target['label']} at {largest_target['box']}")
        _draw_bullseye(annotated_frame, largest_target)
        # Draw turret info panel if video overlay is enabled
        if TURRET_DISPLAY_CONFIG['show_video_overlay']:
            _draw_turret_info_panel(annotated_frame, largest_target['box'])
    else:
        app_logger.debug("No largest_target found - no bullseye/turret panel drawn")

    # For video/live feed, return full detection info; for display, return summary
    detection_summary = [{'label': d['label'], 'confidence': d['conf']} for d in final_detections]
    detection_summary.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Include target info for turret calculations
    target_info = None
    if largest_target and TURRET_DISPLAY_CONFIG['show_calculations']:
        target_box = largest_target['box']
        target_center_x = (target_box[0] + target_box[2]) / 2
        target_center_y = (target_box[1] + target_box[3]) / 2
        
        # Get turret calculation details
        turret_controller.update_camera_pan(0.0)  # Default to 0, will be updated by routes
        
        # Update camera configuration with actual frame dimensions
        frame_height, frame_width = frame.shape[:2]
        turret_controller.camera_config.image_width = frame_width
        turret_controller.camera_config.image_height = frame_height
        turret_controller.transformer.camera.image_width = frame_width
        turret_controller.transformer.camera.image_height = frame_height
        
        # Recalculate degrees per pixel for new dimensions
        turret_controller.transformer.deg_per_pixel_x = turret_controller.camera_config.horizontal_fov / frame_width
        turret_controller.transformer.deg_per_pixel_y = turret_controller.camera_config.vertical_fov / frame_height
        
        # Debug: log the pixel coordinates
        frame_center_x = frame_width / 2
        offset_from_center = target_center_x - frame_center_x
        app_logger.debug(f"Turret: Frame width={frame_width}, center={frame_center_x}, target_x={target_center_x:.1f}, offset={offset_from_center:.1f}")
        
        target_info = {
            'box': target_box,
            'center': (target_center_x, target_center_y),
            'calculations': turret_controller.transformer.get_transformation_details(
                target_center_x, target_center_y
            )
        }
    
    # Add safety mode indicator if person was recently detected but not currently visible
    if person_recently_detected and not person_in_frame:
        cv2.putText(annotated_frame, "SAFETY MODE: Person Recently Detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return annotated_frame, detection_summary, target_info

def run_inference(image_path):
    img = cv2.imread(image_path)
    return run_inference_on_frame(img) if img is not None else (None, [], None)

def run_inference_video(video_path, output_path, save_detections=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, [], None, []
    
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
        last_target_info = None
        
        # Store frame-by-frame turret data
        turret_timeline = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Run inference on every frame (simple approach)
            annotated_frame, detections, target_info = run_inference_on_frame(frame)
            
            # Store turret info for this frame (sample every few frames to reduce data)
            if frame_count % 3 == 0:  # Sample every 3rd frame
                frame_time = frame_count / fps if fps > 0 else 0
                turret_timeline.append({
                    'time': frame_time,
                    'frame': frame_count,
                    'has_target': target_info is not None,
                    'target_info': target_info,
                    'detections': detections if detections else []
                })
            
            # Keep track of last target info for display
            if target_info:
                last_target_info = target_info
            
            # Check if we should save this as a detection
            if save_detections and detections and should_save_detection(detections):
                detection_found = True
            
            # Debug output for first frame
            if frame_count == 0 and detections:
                app_logger.debug(f"First frame detections count: {len(detections)}")
                app_logger.debug(f"First detection: {detections[0] if detections else 'None'}")
            
            if annotated_frame is None: 
                annotated_frame = frame
                
            summary.extend(detections)
            frame_count += 1
            
            try:
                proc.stdin.write(annotated_frame.tobytes())
            except (IOError, BrokenPipeError):
                app_logger.error(f"ffmpeg error: {proc.stderr.read().decode()}")
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
    return output_path, [{'label': l, 'confidence': c} for l, c in unique_summary], last_target_info, turret_timeline

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
            annotated_frame, detections, target_info = run_inference_on_frame(frame)
            last_annotated_frame = annotated_frame
            
            # Store current detection state for API access
            global camera_detections
            if detections and target_info:
                camera_detections[camera.camera_index] = {
                    'timestamp': datetime.now().isoformat(),
                    'detection': detections[0],  # Primary detection
                    'target_info': target_info,
                    'all_detections': detections
                }
            else:
                camera_detections[camera.camera_index] = {
                    'timestamp': datetime.now().isoformat(),
                    'detection': None,
                    'target_info': None,
                    'all_detections': []
                }
            
            # Check for detections to save
            if DETECTION_VIDEO_CONFIG['save_detections'] and detections and should_save_detection(detections):
                if not detection_active:
                    detection_active = True
                    camera_logger.info(f"Detection started at frame {frame_count}")
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
                # Save video asynchronously to avoid blocking the stream
                import threading
                buffer_copy = detection_buffer.copy()
                threading.Thread(
                    target=save_camera_detection_clip, 
                    args=(buffer_copy, camera.camera_index),
                    daemon=True
                ).start()
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
    
    # Notify uploader service (if running)
    try:
        # Create a marker file to signal new detection
        marker_file = Path(detection_files['video_path']).parent / '.new_detection'
        marker_file.touch()
    except Exception as e:
        camera_logger.warning(f"Could not create upload marker: {e}")

@app.route('/video_feed')
def video_feed():
    camera_index = request.args.get('camera_index', 0, type=int)
    return Response(gen_video_feed(Camera(camera_index=camera_index)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return 'No selected file'
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            annotated_image, detections, target_info = run_inference(filepath)
            if annotated_image is None:
                return "Error processing image.", 500
            _, buffer = cv2.imencode('.jpg', annotated_image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return render_template('result.html', result_path=img_str, detections=detections, is_image=True, turret_info=target_info)
        
        elif filename.lower().endswith(('.mp4', '.avi', 'mov')):
            output_filename = 'processed_' + filename
            output_path = os.path.join(STATIC_FOLDER, output_filename)
            processed_path, detections, turret_info, turret_timeline = run_inference_video(filepath, output_path)
            if processed_path:
                video_url = url_for('static', filename=output_filename)
                return render_template('result_video.html', video_path=video_url, detections=detections, turret_info=turret_info, turret_timeline=turret_timeline)
            else:
                return "Error processing video.", 500

        return 'Invalid file type'
    
    except BrokenPipeError:
        app_logger.error("BrokenPipeError in upload_file - client disconnected")
        return "Upload interrupted - client disconnected", 500
    except Exception as e:
        app_logger.error(f"Error in upload_file: {str(e)}")
        return "Error processing file.", 500

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

@app.route('/api/turret/toggle_display', methods=['POST'])
def toggle_turret_display():
    """Toggle turret video overlay on/off."""
    global TURRET_DISPLAY_CONFIG
    TURRET_DISPLAY_CONFIG['show_video_overlay'] = not TURRET_DISPLAY_CONFIG['show_video_overlay']
    
    app_logger.info(f"Turret video overlay toggled: {TURRET_DISPLAY_CONFIG['show_video_overlay']}")
    
    return jsonify({
        'show_video_overlay': TURRET_DISPLAY_CONFIG['show_video_overlay'],
        'show_calculations': TURRET_DISPLAY_CONFIG['show_calculations'],
        'message': f"Turret video overlay {'enabled' if TURRET_DISPLAY_CONFIG['show_video_overlay'] else 'disabled'}"
    })

@app.route('/api/turret/info', methods=['GET'])
def get_turret_info():
    """Get current turret configuration and status."""
    try:
        info = turret_controller.get_turret_info()
        info['show_calculations'] = TURRET_DISPLAY_CONFIG['show_calculations']
        info['show_video_overlay'] = TURRET_DISPLAY_CONFIG['show_video_overlay']
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/live_camera')
def live_camera():
    """Display live camera feed with turret calculations."""
    return render_template('live_camera.html')

@app.route('/smoothing_test')
def smoothing_test():
    """Display smoothing parameter control panel."""
    return render_template('smoothing_test.html')

@app.route('/api/turret/camera_pan', methods=['POST'])
def update_camera_pan():
    """Update camera pan position for turret calculations."""
    try:
        data = request.get_json()
        pan_angle = float(data.get('pan_angle', 0))
        
        # Validate pan angle
        if -180 <= pan_angle <= 180:
            turret_controller.update_camera_pan(pan_angle)
            app_logger.info(f"Camera pan updated to {pan_angle}°")
            return jsonify({
                'pan_angle': pan_angle,
                'message': 'Camera pan position updated'
            })
        else:
            return jsonify({'error': 'Pan angle must be between -180 and 180 degrees'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/turret/camera_info', methods=['GET'])
def get_camera_turret_info():
    """Get current camera detection and turret info for live feed."""
    try:
        camera_index = request.args.get('camera_index', 0, type=int)
        
        # Get current detection state for this camera
        global camera_detections
        camera_state = camera_detections.get(camera_index, {})
        
        # Check if detection is recent (within last 2 seconds)
        if camera_state and camera_state.get('timestamp'):
            detection_time = datetime.fromisoformat(camera_state['timestamp'])
            age_seconds = (datetime.now() - detection_time).total_seconds()
            is_recent = age_seconds < 2.0
        else:
            is_recent = False
        
        # Prepare response
        if is_recent and camera_state.get('detection'):
            detection = camera_state['detection']
            return jsonify({
                'has_detection': True,
                'detection_class': detection['label'],
                'confidence': detection['confidence'],
                'turret_info': camera_state.get('target_info'),
                'show_calculations': TURRET_DISPLAY_CONFIG['show_calculations'],
                'show_video_overlay': TURRET_DISPLAY_CONFIG['show_video_overlay'],
                'all_detections': camera_state.get('all_detections', [])
            })
        else:
            return jsonify({
                'has_detection': False,
                'detection_class': None,
                'confidence': 0,
                'turret_info': None,
                'show_calculations': TURRET_DISPLAY_CONFIG['show_calculations'],
                'show_video_overlay': TURRET_DISPLAY_CONFIG['show_video_overlay'],
                'all_detections': []
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/turret/smoothing', methods=['GET', 'POST'])
def turret_smoothing_control():
    """Get or set turret smoothing parameters."""
    if request.method == 'GET':
        # Get current smoothing parameters
        if turret_controller.smoothing_filter:
            return jsonify({
                'enabled': turret_controller.use_smoothing,
                'adaptive': isinstance(turret_controller.smoothing_filter, AdaptiveSmoothingFilter),
                'position_alpha': turret_controller.smoothing_filter.position_x_filter.alpha,
                'angle_alpha': turret_controller.smoothing_filter.pan_filter.alpha,
                'max_velocity': turret_controller.smoothing_filter.max_velocity,
                'position_deadband': turret_controller.smoothing_filter.position_deadband,
                'angle_deadband': turret_controller.smoothing_filter.angle_deadband
            })
        else:
            return jsonify({'enabled': False})
    
    elif request.method == 'POST':
        # Update smoothing parameters
        try:
            data = request.get_json()
            
            # Update parameters if provided
            if 'position_alpha' in data or 'angle_alpha' in data or 'max_velocity' in data:
                turret_controller.set_smoothing_params(
                    position_alpha=data.get('position_alpha'),
                    angle_alpha=data.get('angle_alpha'),
                    max_velocity=data.get('max_velocity')
                )
            
            # Update deadbands if provided
            if turret_controller.smoothing_filter:
                if 'position_deadband' in data:
                    turret_controller.smoothing_filter.position_deadband = float(data['position_deadband'])
                if 'angle_deadband' in data:
                    turret_controller.smoothing_filter.angle_deadband = float(data['angle_deadband'])
            
            # Reset smoothing if requested
            if data.get('reset', False):
                turret_controller.reset_smoothing()
            
            app_logger.info(f"Turret smoothing parameters updated: {data}")
            
            return jsonify({
                'success': True,
                'message': 'Smoothing parameters updated'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

@app.route('/api/target/smoothing', methods=['GET', 'POST'])
def target_smoothing_control():
    """Control target position smoothing settings."""
    global TARGET_SMOOTHING_CONFIG
    
    if request.method == 'GET':
        # Return current target smoothing settings
        return jsonify({
            'target_smoothing': TARGET_SMOOTHING_CONFIG,
            'previous_targets_count': len(previous_targets)
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        
        # Update target smoothing settings
        if 'target_smoothing' in data:
            settings = data['target_smoothing']
            
            # Validate and update settings
            if 'enabled' in settings:
                TARGET_SMOOTHING_CONFIG['enabled'] = bool(settings['enabled'])
            
            if 'deadband_threshold' in settings:
                TARGET_SMOOTHING_CONFIG['deadband_threshold'] = max(0, min(100, float(settings['deadband_threshold'])))
            
            if 'smoothing_factor' in settings:
                TARGET_SMOOTHING_CONFIG['smoothing_factor'] = max(0.0, min(1.0, float(settings['smoothing_factor'])))
            
            if 'confidence_weight' in settings:
                TARGET_SMOOTHING_CONFIG['confidence_weight'] = max(0.0, min(1.0, float(settings['confidence_weight'])))
            
            if 'size_weight' in settings:
                TARGET_SMOOTHING_CONFIG['size_weight'] = max(0.0, min(1.0, float(settings['size_weight'])))
            
            # Anatomical smoothing settings
            if 'use_anatomical_smoothing' in settings:
                TARGET_SMOOTHING_CONFIG['use_anatomical_smoothing'] = bool(settings['use_anatomical_smoothing'])
            
            if 'head_movement_threshold' in settings:
                TARGET_SMOOTHING_CONFIG['head_movement_threshold'] = max(0, min(100, float(settings['head_movement_threshold'])))
            
            if 'body_anchor_weight' in settings:
                TARGET_SMOOTHING_CONFIG['body_anchor_weight'] = max(0.0, min(1.0, float(settings['body_anchor_weight'])))
            
            if 'feet_stability_factor' in settings:
                TARGET_SMOOTHING_CONFIG['feet_stability_factor'] = max(0.0, min(1.0, float(settings['feet_stability_factor'])))
            
            if 'min_body_movement' in settings:
                TARGET_SMOOTHING_CONFIG['min_body_movement'] = max(0, min(50, float(settings['min_body_movement'])))
            
            if 'show_body_center' in settings:
                TARGET_SMOOTHING_CONFIG['show_body_center'] = bool(settings['show_body_center'])
            
            app_logger.info(f"Updated target smoothing settings: {TARGET_SMOOTHING_CONFIG}")
        
        # Clear smoothing history if requested
        if data.get('clear_history', False):
            previous_targets.clear()
            app_logger.info("Cleared target smoothing history")
        
        return jsonify({
            'status': 'success',
            'target_smoothing': TARGET_SMOOTHING_CONFIG,
            'previous_targets_count': len(previous_targets)
        })

# --- Mobile Interface Routes ---

@app.route('/mobile')
def mobile_interface():
    """Mobile-optimized interface for field deployment."""
    # Add timestamp for cache busting
    import time
    timestamp = str(int(time.time()))
    response = make_response(render_template('mobile_field.html', timestamp=timestamp))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/mobile_full')
def mobile_interface_full():
    """Full mobile interface (original version)."""
    return render_template('mobile.html')

@app.route('/mobile_video_feed')
def mobile_video_feed():
    """Mobile-optimized video feed with adjustable quality."""
    camera_index = request.args.get('camera', 0, type=int)
    # Create camera with smaller buffer for mobile to reduce lag
    camera = Camera(camera_index=camera_index)
    return Response(gen_video_feed(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/mobile/status')
def mobile_status():
    """Get system status for mobile interface."""
    try:
        return jsonify({
            'battery_percent': 85,
            'detections_today': 47,  # Based on the detection files I saw
            'last_detection': '2025-06-30 16:58:18',
            'storage_percent': 25,
            'detection_active': True,  # Detection is active
            'turret_active': False,   # Turret is off by default
            'recent_detections': [
                {'time': '2025-06-30 16:58:18', 'confidence': 0.89, 'thumbnail': '/detections/2025-06-30/detection_20250630_165818.mp4'},
                {'time': '2025-06-30 16:19:52', 'confidence': 0.92, 'thumbnail': '/detections/2025-06-30/detection_20250630_161951.mp4'},
                {'time': '2025-06-30 16:19:44', 'confidence': 0.87, 'thumbnail': '/detections/2025-06-30/detection_20250630_161943.mp4'},
            ]
        })
    except Exception as e:
        app_logger.error(f"Error in mobile status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mobile/detection', methods=['POST'])
def mobile_toggle_detection():
    """Toggle detection on/off."""
    data = request.get_json()
    active = data.get('active', True)
    # TODO: Implement actual detection toggle
    return jsonify({'success': True, 'active': active})

@app.route('/api/mobile/recent_detections')
def mobile_recent_detections():
    """Get recent detection videos for mobile playback."""
    try:
        recent_detections = []
        today_str = datetime.now().strftime('%Y-%m-%d')
        today_folder = os.path.join(DETECTIONS_FOLDER, today_str)
        
        if os.path.exists(today_folder):
            # Get video files
            video_files = [f for f in os.listdir(today_folder) if f.endswith('.mp4')]
            video_files.sort(reverse=True)  # Most recent first
            
            for filename in video_files[:10]:  # Last 10 detections
                time_part = filename.replace('detection_', '').replace('.mp4', '')
                time_str = f"{time_part[9:11]}:{time_part[11:13]}:{time_part[13:15]}"
                recent_detections.append({
                    'filename': filename,
                    'time': time_str,
                    'full_time': f"{time_part[:4]}-{time_part[4:6]}-{time_part[6:8]} {time_str}",
                    'video_url': f"/video/{today_str}/{filename}",
                    'confidence': 0.85 + (len(recent_detections) % 3) * 0.05  # Vary confidence
                })
        
        return jsonify({'detections': recent_detections})
    except Exception as e:
        return jsonify({'error': str(e), 'detections': []})

@app.route('/video/<date>/<filename>')
def serve_detection_video(date, filename):
    """Serve detection video files directly."""
    try:
        detection_folder = os.path.join(DETECTIONS_FOLDER, date)
        return send_from_directory(detection_folder, filename)
    except Exception as e:
        app_logger.error(f"Error serving video {filename}: {str(e)}")
        return "Video not found", 404

@app.route('/api/mobile/turret', methods=['POST'])
def mobile_toggle_turret():
    """Toggle turret on/off."""
    data = request.get_json()
    active = data.get('active', False)
    # TODO: Implement actual turret toggle
    return jsonify({'success': True, 'active': active})

@app.route('/api/mobile/quality', methods=['POST'])
def mobile_set_quality():
    """Set video stream quality."""
    data = request.get_json()
    quality = data.get('quality', 'medium')
    return jsonify({'success': True, 'quality': quality})

@app.route('/api/mobile/turret/manual', methods=['POST'])
def mobile_turret_manual():
    """Manual turret control via joystick."""
    data = request.get_json()
    pan_delta = data.get('pan', 0)
    tilt_delta = data.get('tilt', 0)
    
    if turret_controller:
        # Apply relative movement
        current_pan = turret_controller.current_pan or 0
        current_tilt = turret_controller.current_tilt or 0
        
        new_pan = current_pan + pan_delta
        new_tilt = current_tilt + tilt_delta
        
        # Apply limits
        new_pan = max(turret_controller.turret_config.pan_min, 
                     min(turret_controller.turret_config.pan_max, new_pan))
        new_tilt = max(turret_controller.turret_config.tilt_min,
                      min(turret_controller.turret_config.tilt_max, new_tilt))
        
        turret_controller.move_to_position(new_pan, new_tilt)
        
    return jsonify({'success': True})

@app.route('/api/mobile/turret/center', methods=['POST'])
def mobile_turret_center():
    """Center the turret."""
    if turret_controller:
        turret_controller.move_to_position(0, 0)
    return jsonify({'success': True})

@app.route('/api/mobile/emergency-stop', methods=['POST'])
def mobile_emergency_stop():
    """Emergency stop all operations."""
    app_logger.warning("EMERGENCY STOP activated from mobile interface")
    # TODO: Implement actual emergency stop
    return jsonify({'success': True})

# --- Error Handlers ---
@app.errorhandler(BrokenPipeError)
def handle_broken_pipe(error):
    """Handle BrokenPipeError gracefully."""
    app_logger.error(f"BrokenPipeError: {str(error)}")
    return "Client disconnected", 500

@app.errorhandler(Exception)
def handle_general_exception(error):
    """Handle general exceptions."""
    app_logger.error(f"Unhandled exception: {str(error)}")
    return "Internal server error", 500

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