import cv2
from ultralytics import YOLO
import os
import subprocess
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.math_utils import calculate_iou
from shared.model_manager import model_manager
from shared.model_config_loader import model_config

# --- Constants and Model Loading ---
# Load model paths from configuration
DEER_MODEL_PATH = model_config.get_model_path('deer')
GENERAL_MODEL_PATH = model_config.get_model_path('general')

# Fall back to default paths if config not found
if not DEER_MODEL_PATH:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    DEER_MODEL_PATH = os.path.join(_script_dir, 'models', 'best.pt')
if not GENERAL_MODEL_PATH:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    GENERAL_MODEL_PATH = os.path.join(_script_dir, 'models', 'yolov8n.pt')

# Use singleton model manager to avoid duplicate loading
deer_model = model_manager.get_deer_model(DEER_MODEL_PATH)
general_model = model_manager.get_general_model(GENERAL_MODEL_PATH)

# A set of common animal classes from the COCO dataset for easy lookup
ANIMAL_CLASSES = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}

# --- Drawing Helpers ---

def _draw_bounding_box(frame, box, label, conf, color):
    """Draws a single bounding box and label on the frame."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_text = f"{label} {conf:.2f}"
    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def _draw_bullseye(frame, target):
    """Draws a bullseye on the center of a target detection."""
    box = target['box']
    center_x = int((box[0] + box[2]) / 2)
    center_y = int((box[1] + box[3]) / 2)
    # Draw a red bullseye
    cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), 2)
    cv2.circle(frame, (center_x, center_y), 20, (0, 0, 255), 2)
    cv2.line(frame, (center_x - 25, center_y), (center_x + 25, center_y), (0, 0, 255), 2)
    cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 25), (0, 0, 255), 2)

# --- Core Inference Logic ---

def run_inference_on_frame(frame, conf_threshold=None):
    """
    Runs full inference and annotation logic on a single frame.
    - Detects deer and general objects.
    - Filters overlapping detections.
    - Implements safety logic: if a person is present, animals are "Pets" and not targeted.
    - If no person is present, targets the largest animal.
    - Returns the annotated frame and a summary of detections.
    """
    # Use confidence thresholds from configuration if not specified
    deer_conf = conf_threshold or model_config.get_confidence_threshold('deer')
    general_conf = conf_threshold or model_config.get_confidence_threshold('general')
    
    # Check if dual model is enabled
    use_dual_model = model_config.is_dual_model_enabled()
    safety_mode = model_config.is_safety_mode_enabled()
    
    # 1. Get all detections from both models
    deer_results = deer_model(frame, verbose=False, conf=deer_conf)
    general_results = general_model(frame, verbose=False, conf=general_conf) if use_dual_model else []

    deer_detections = []
    for r in deer_results:
        if r.boxes is not None:
            deer_detections.extend([
                {'box': b.xyxy[0].tolist(), 'conf': float(b.conf[0]), 'cls': int(b.cls[0]), 'label': deer_model.names[int(b.cls[0])]}
                for b in r.boxes if b.conf[0] > conf_threshold
            ])
    
    general_detections = []
    for r in general_results:
        if r.boxes is not None:
            general_detections.extend([
                {'box': b.xyxy[0].tolist(), 'conf': float(b.conf[0]), 'cls': int(b.cls[0]), 'label': general_model.names[int(b.cls[0])]}
                for b in r.boxes if b.conf[0] > conf_threshold
            ])

    # 2. Filter general detections that overlap with deer detections (but always keep people)
    final_detections = list(deer_detections)
    for gen_det in general_detections:
        is_overlapping = False
        if gen_det['label'] != 'person':
            for deer_det in deer_detections:
                if calculate_iou(gen_det['box'], deer_det['box']) > 0.5:
                    is_overlapping = True
                    break
        if not is_overlapping:
            final_detections.append(gen_det)

    # 3. Apply safety and targeting logic
    person_present = any(d['label'] == 'person' for d in final_detections) if safety_mode else False
    largest_target = None

    if not person_present:
        largest_area = 0
        for det in final_detections:
            # Any animal can be a target if no person is present
            if det['label'] == 'deer' or det['label'] in ANIMAL_CLASSES:
                box = det['box']
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > largest_area:
                    largest_area = area
                    largest_target = det

    # 4. Draw annotations
    annotated_frame = frame.copy()
    for det in final_detections:
        label = det['label']

        if person_present:
            if det['label'] == 'person':
                color = (0, 255, 0)  # Green for person
            elif det['label'] == 'deer' or det['label'] in ANIMAL_CLASSES:
                label = 'Pet'
                color = (255, 0, 0)  # Blue for pets (when person present)
        else: # No person
            if det['label'] == 'deer' or det['label'] in ANIMAL_CLASSES:
                color = (0, 0, 255)  # Red for all animals (when no person)
            elif det['label'] == 'person':
                color = (0, 255, 0)  # Green for person
            else:
                color = (128, 128, 128)  # Gray for other objects

        _draw_bounding_box(annotated_frame, det['box'], label, det['conf'], color)

    if largest_target:
        _draw_bullseye(annotated_frame, largest_target)

    # 5. Create summary for UI
    detection_summary = [{'label': d['label'], 'confidence': d['conf']} for d in final_detections]
    detection_summary.sort(key=lambda x: x['confidence'], reverse=True)

    return annotated_frame, detection_summary

# --- Wrappers for Image and Video Processing ---

def run_inference(image_path):
    """Wrapper to run inference on a single image file."""
    img = cv2.imread(image_path)
    if img is None:
        return None, []
    return run_inference_on_frame(img)

def run_inference_video(video_path, output_path):
    """Wrapper to process a video file frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video at {video_path}")
        return None, []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_path
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    all_detections_summary = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, detections = run_inference_on_frame(frame)
        if annotated_frame is None: # Handle case where frame processing might fail
            continue
            
        all_detections_summary.extend(detections)

        try:
            process.stdin.write(annotated_frame.tobytes())
        except (IOError, BrokenPipeError):
            logging.error(f"ffmpeg process ended unexpectedly. Error: {process.stderr.read().decode()}")
            break
            
    cap.release()
    if process.stdin:
        process.stdin.close()
    process.wait()
    
    unique_detections = { (d.get('label', 'N/A'), f"{d.get('confidence', 0):.2f}") for d in all_detections_summary }
    sorted_detections = sorted(list(unique_detections), key=lambda x: float(x[1]), reverse=True)
    final_summary = [{'label': label, 'confidence': conf} for label, conf in sorted_detections]

    return output_path, final_summary

