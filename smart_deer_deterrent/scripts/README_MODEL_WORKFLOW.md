# Model Management Workflow

This directory contains scripts for managing the deer detection models used in the Smart Deer Deterrent System.

## Overview

The system uses a dual-model approach:
- **Deer Model**: Specialized model trained specifically for deer detection
- **General Model**: YOLOv8n for detecting people and other animals

## Scripts

### 1. `update_model.py` - Model Update Tool

Transfer trained models from your training project to deployment.

```bash
# Update to latest model
python scripts/update_model.py

# List all model versions
python scripts/update_model.py --list

# Rollback to a previous version
python scripts/update_model.py --rollback v2_20240630_143022

# Force update even if model exists
python scripts/update_model.py --force

# Specify custom paths
python scripts/update_model.py \
    --training-path "/path/to/training/project" \
    --deployment-path "/path/to/deployment/project"
```

### 2. `evaluate_model.py` - Model Evaluation Tool

Test model performance before deployment.

```bash
# Evaluate on a single image
python scripts/evaluate_model.py /path/to/test/image.jpg

# Evaluate on a directory of images
python scripts/evaluate_model.py /path/to/test/images/
python scripts/evaluate_model.py /path/to/test/images/ --save-outputs

# Evaluate on video
python scripts/evaluate_model.py /path/to/test/video.mp4 --video-sample-rate 30

# Compare two models
python scripts/evaluate_model.py /path/to/test/images/ \
    --compare-with /path/to/other/model.pt

# Use custom confidence threshold
python scripts/evaluate_model.py /path/to/test/images/ --confidence 0.5
```

## Configuration

Model settings are stored in `config/model_config.json`:

```json
{
  "models": {
    "deer": {
      "path": "camera_detection/models/best.pt",
      "confidence_threshold": 0.3,
      "current_version": "best.pt"
    },
    "general": {
      "path": "camera_detection/models/yolov8n.pt",
      "confidence_threshold": 0.25
    }
  },
  "inference_settings": {
    "use_dual_model": true,
    "safety_mode": true,
    "device": "auto"
  }
}
```

## Recommended Workflow

### 1. Training a New Model

In your training project:
```bash
cd "/Users/gsknight/Documents/deer detection training"
python train_deer_model.py --epochs 100 --batch-size 16
```

### 2. Evaluating the New Model

Test the model before deployment:
```bash
# In the deployment project
python scripts/evaluate_model.py \
    "/Users/gsknight/Documents/deer detection training/test_images" \
    --model "/Users/gsknight/Documents/deer detection training/runs/detect/train/weights/best.pt"
```

### 3. Updating the Deployment

If satisfied with performance:
```bash
python scripts/update_model.py
```

### 4. Testing in Application

Test with the web interface:
1. Start the Flask app: `python main.py`
2. Upload test videos/images
3. Monitor detection accuracy

### 5. Rollback if Needed

If issues are found:
```bash
# List versions
python scripts/update_model.py --list

# Rollback to previous version
python scripts/update_model.py --rollback v1_20240629_120000
```

## Model Version Tracking

The system automatically tracks:
- Model version numbers
- Upload timestamps
- File checksums (to prevent duplicates)
- Performance metrics (if available)
- Active model version

Version history is stored in `camera_detection/models/model_versions.json`.

## Tips for Better Models

1. **Dataset Quality**
   - Include diverse lighting conditions (dawn, day, dusk, night)
   - Various deer poses and distances
   - Different backgrounds and environments
   - Edge cases (partial visibility, groups)

2. **Training Parameters**
   - Start with YOLOv8n for speed, upgrade to s/m for accuracy
   - Use data augmentation for robustness
   - Monitor validation metrics during training
   - Use early stopping to prevent overfitting

3. **Testing**
   - Always test on real-world data before deployment
   - Check both detection rate and false positive rate
   - Test in conditions matching deployment environment
   - Verify performance at different times of day

4. **Configuration Tuning**
   - Adjust confidence thresholds based on your needs
   - Lower threshold = more detections but more false positives
   - Higher threshold = fewer false positives but might miss detections
   - Test thoroughly after any configuration changes