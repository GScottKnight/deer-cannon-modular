import base64
import os
import subprocess
import cv2
from flask import Flask, render_template, request, jsonify, Response, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(_script_dir, 'uploads')
STATIC_FOLDER = os.path.join(_script_dir, 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model Loading ---
DEER_MODEL_PATH = os.path.join(_script_dir, 'models', 'deer_model.pt')
GENERAL_MODEL_PATH = os.path.join(_script_dir, 'models', 'yolov8n.pt')
deer_model = YOLO(DEER_MODEL_PATH)
general_model = YOLO(GENERAL_MODEL_PATH)

# A set of common animal classes from the COCO dataset for easy lookup
ANIMAL_CLASSES = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}

# --- Utility Functions ---

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return 0.0 if denominator == 0 else interArea / denominator

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
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            cap.release()
    return indices

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.video = cv2.VideoCapture(self.camera_index)
        if not self.video.isOpened():
            raise RuntimeError("Could not start camera.")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        return frame if success else None

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

def run_inference_on_frame(frame):
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

    # Filter overlapping detections within the deer model first (less aggressive - only filter true duplicates)
    deer_detections = filter_overlapping_detections(deer_detections, iou_threshold=0.8)
    
    # Filter overlapping detections within the general model (less aggressive)
    general_detections = filter_overlapping_detections(general_detections, iou_threshold=0.8)

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

    person_present = any(d['label'] == 'person' for d in final_detections)
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
        _draw_bounding_box(annotated_frame, det['box'], label, det['conf'], color)

    if largest_target:
        _draw_bullseye(annotated_frame, largest_target)

    detection_summary = [{'label': d['label'], 'confidence': d['conf']} for d in final_detections]
    detection_summary.sort(key=lambda x: x['confidence'], reverse=True)
    return annotated_frame, detection_summary

def run_inference(image_path):
    img = cv2.imread(image_path)
    return run_inference_on_frame(img) if img is not None else (None, [])

def run_inference_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, []
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_path]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    summary = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        annotated_frame, detections = run_inference_on_frame(frame)
        if annotated_frame is None: continue
        summary.extend(detections)
        try:
            proc.stdin.write(annotated_frame.tobytes())
        except (IOError, BrokenPipeError):
            print(f"ffmpeg error: {proc.stderr.read().decode()}")
            break
    cap.release()
    if proc.stdin:
        proc.stdin.close()
    proc.wait()
    unique_summary = sorted(list({(d['label'], f"{d['confidence']:.2f}") for d in summary}), key=lambda x: float(x[1]), reverse=True)
    return output_path, [{'label': l, 'confidence': c} for l, c in unique_summary]

# --- Flask Routes ---

@app.route('/cameras')
def list_cameras():
    return jsonify(get_available_cameras())

@app.route('/')
def index():
    return render_template('index.html')

def gen_video_feed(camera):
    while True:
        frame = camera.get_frame()
        if frame is None: break
        annotated_frame, _ = run_inference_on_frame(frame)
        (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

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

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 