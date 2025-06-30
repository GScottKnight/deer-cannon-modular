import cv2
from ultralytics import YOLO
import os
import subprocess
from smart_deer_deterrent.shared.math_utils import calculate_iou

# --- Constants and Model Loading ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
DEER_MODEL_PATH = os.path.join(_script_dir, 'models', 'deer_model.pt')
GENERAL_MODEL_PATH = os.path.join(_script_dir, 'models', 'yolov8n.pt')

# Load the models
deer_model = YOLO(DEER_MODEL_PATH)
general_model = YOLO(GENERAL_MODEL_PATH)

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

def run_inference_on_frame(frame):
    """
    Runs full inference and annotation logic on a single frame.
    - Detects deer and general objects.
    - Filters overlapping detections.
    - Implements safety logic: if a person is present, animals are "Pets" and not targeted.
    - If no person is present, targets the largest animal.
    - Returns the annotated frame and a summary of detections.
    """
    # 1. Get all detections from both models
    deer_results = deer_model(frame, verbose=False)
    general_results = general_model(frame, verbose=False)

    deer_detections = [
        {'box': b.xyxy[0].tolist(), 'conf': float(b.conf[0]), 'cls': int(b.cls[0]), 'label': deer_model.names[int(b.cls[0])]}
        for r in deer_results for b in r.boxes if b.conf[0] > 0.5
    ]
    general_detections = [
        {'box': b.xyxy[0].tolist(), 'conf': float(b.conf[0]), 'cls': int(b.cls[0]), 'label': general_model.names[int(b.cls[0])]}
        for r in general_results for b in r.boxes if b.conf[0] > 0.5
    ]

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
    person_present = any(d['label'] == 'person' for d in final_detections)
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
        label, color = det['label'], (255, 0, 0) # Default to blue for general objects

        if person_present:
            if det['label'] == 'person':
                color = (0, 255, 0)  # Green
            elif det['label'] == 'deer' or det['label'] in ANIMAL_CLASSES:
                label = 'Pet'
                color = (255, 0, 0)  # Blue
        else: # No person
            if det['label'] == 'deer':
                color = (0, 0, 255) # Red
            elif det['label'] in ANIMAL_CLASSES:
                 color = (255, 0, 0) # Blue for other animals

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
        print(f"Error: Could not open video at {video_path}")
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
            print(f"ffmpeg process ended unexpectedly. Error: {process.stderr.read().decode()}")
            break
            
    cap.release()
    if process.stdin:
        process.stdin.close()
    process.wait()
    
    unique_detections = { (d.get('label', 'N/A'), f"{d.get('confidence', 0):.2f}") for d in all_detections_summary }
    sorted_detections = sorted(list(unique_detections), key=lambda x: float(x[1]), reverse=True)
    final_summary = [{'label': label, 'confidence': conf} for label, conf in sorted_detections]

    return output_path, final_summary

