#!/usr/bin/env python3
"""
Model Evaluation Script for Smart Deer Deterrent System

This script evaluates the performance of the deployed model
on test images or videos to ensure quality before deployment.
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from shared.model_manager import model_manager


class ModelEvaluator:
    def __init__(self, model_path=None):
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Use the deployed model
            model_path = Path(__file__).parent.parent / "camera_detection" / "models" / "best.pt"
            self.model = model_manager.get_deer_model(str(model_path))
        
        self.results = {
            "total_images": 0,
            "total_detections": 0,
            "detections_by_confidence": defaultdict(int),
            "processing_times": [],
            "images_with_detections": 0,
            "confidence_scores": [],
            "evaluation_date": datetime.now().isoformat(),
            "model_path": str(model_path)
        }
    
    def evaluate_image(self, image_path, confidence_threshold=0.3, save_output=False):
        """Evaluate model on a single image."""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return None
        
        # Run inference
        start_time = cv2.getTickCount()
        results = self.model(img, conf=confidence_threshold, verbose=False)
        end_time = cv2.getTickCount()
        
        # Calculate processing time
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        self.results["processing_times"].append(processing_time)
        
        # Process detections
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    self.results["confidence_scores"].append(conf)
                    
                    # Categorize by confidence level
                    if conf >= 0.8:
                        self.results["detections_by_confidence"]["high"] += 1
                    elif conf >= 0.5:
                        self.results["detections_by_confidence"]["medium"] += 1
                    else:
                        self.results["detections_by_confidence"]["low"] += 1
                    
                    detections.append({
                        "confidence": conf,
                        "box": box.xyxy[0].tolist(),
                        "class": self.model.names[int(box.cls[0])]
                    })
        
        # Update statistics
        self.results["total_images"] += 1
        self.results["total_detections"] += len(detections)
        if detections:
            self.results["images_with_detections"] += 1
        
        # Save annotated image if requested
        if save_output and detections:
            output_dir = Path("evaluation_outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Draw results
            annotated = results[0].plot()
            output_path = output_dir / f"eval_{Path(image_path).name}"
            cv2.imwrite(str(output_path), annotated)
        
        return {
            "image": str(image_path),
            "detections": detections,
            "processing_time": processing_time
        }
    
    def evaluate_directory(self, directory_path, confidence_threshold=0.3, save_outputs=False):
        """Evaluate model on all images in a directory."""
        directory = Path(directory_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        print(f"📁 Found {len(image_files)} images to evaluate")
        
        all_results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {img_path.name}", end='\r')
            result = self.evaluate_image(img_path, confidence_threshold, save_outputs)
            if result:
                all_results.append(result)
        
        print("\n✅ Evaluation complete!")
        
        return all_results
    
    def evaluate_video(self, video_path, confidence_threshold=0.3, sample_rate=30):
        """Evaluate model on video frames."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        
        frame_count = 0
        results_by_frame = []
        
        print(f"📹 Evaluating video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on sample_rate
            if frame_count % sample_rate == 0:
                # Run inference
                start_time = cv2.getTickCount()
                results = self.model(frame, conf=confidence_threshold, verbose=False)
                end_time = cv2.getTickCount()
                
                processing_time = (end_time - start_time) / cv2.getTickFrequency()
                self.results["processing_times"].append(processing_time)
                
                # Count detections
                detection_count = 0
                for r in results:
                    if r.boxes is not None:
                        detection_count = len(r.boxes)
                        for box in r.boxes:
                            self.results["confidence_scores"].append(float(box.conf[0]))
                
                self.results["total_images"] += 1
                self.results["total_detections"] += detection_count
                if detection_count > 0:
                    self.results["images_with_detections"] += 1
                
                results_by_frame.append({
                    "frame": frame_count,
                    "detections": detection_count,
                    "processing_time": processing_time
                })
                
                print(f"Frame {frame_count}: {detection_count} detections", end='\r')
            
            frame_count += 1
        
        cap.release()
        print(f"\n✅ Processed {frame_count} frames (sampled every {sample_rate} frames)")
        
        return results_by_frame
    
    def generate_report(self):
        """Generate evaluation report."""
        if not self.results["total_images"]:
            print("❌ No images evaluated yet!")
            return
        
        # Calculate statistics
        avg_processing_time = np.mean(self.results["processing_times"])
        avg_confidence = np.mean(self.results["confidence_scores"]) if self.results["confidence_scores"] else 0
        detection_rate = self.results["images_with_detections"] / self.results["total_images"] * 100
        
        report = {
            **self.results,
            "statistics": {
                "average_processing_time": f"{avg_processing_time:.3f}s",
                "average_confidence": f"{avg_confidence:.3f}",
                "detection_rate": f"{detection_rate:.1f}%",
                "detections_per_image": self.results["total_detections"] / self.results["total_images"]
            }
        }
        
        # Print report
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION REPORT")
        print("="*60)
        print(f"Model: {Path(self.results['model_path']).name}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📈 Overall Statistics:")
        print(f"  - Total images/frames: {self.results['total_images']}")
        print(f"  - Total detections: {self.results['total_detections']}")
        print(f"  - Detection rate: {detection_rate:.1f}%")
        print(f"  - Avg detections per image: {report['statistics']['detections_per_image']:.2f}")
        
        print(f"\n⚡ Performance:")
        print(f"  - Avg processing time: {avg_processing_time:.3f}s")
        print(f"  - FPS capability: {1/avg_processing_time:.1f}")
        
        print(f"\n🎯 Confidence Distribution:")
        total_dets = self.results["total_detections"]
        if total_dets > 0:
            for level, count in self.results["detections_by_confidence"].items():
                percentage = count / total_dets * 100
                print(f"  - {level.capitalize()}: {count} ({percentage:.1f}%)")
        print(f"  - Average confidence: {avg_confidence:.3f}")
        
        # Save report
        report_path = Path("evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {report_path}")
        
        return report
    
    def compare_models(self, other_model_path, test_images_dir):
        """Compare current model with another model."""
        print(f"\n🔄 Comparing models:")
        print(f"  - Current: {Path(self.results['model_path']).name}")
        print(f"  - Other: {Path(other_model_path).name}")
        
        # Evaluate other model
        other_evaluator = ModelEvaluator(other_model_path)
        other_evaluator.evaluate_directory(test_images_dir)
        
        # Compare results
        current_detection_rate = self.results["images_with_detections"] / self.results["total_images"] * 100
        other_detection_rate = other_evaluator.results["images_with_detections"] / other_evaluator.results["total_images"] * 100
        
        current_avg_time = np.mean(self.results["processing_times"])
        other_avg_time = np.mean(other_evaluator.results["processing_times"])
        
        print(f"\n📊 Comparison Results:")
        print(f"  Detection Rate:")
        print(f"    - Current: {current_detection_rate:.1f}%")
        print(f"    - Other: {other_detection_rate:.1f}%")
        print(f"    - Difference: {current_detection_rate - other_detection_rate:+.1f}%")
        
        print(f"  Processing Speed:")
        print(f"    - Current: {current_avg_time:.3f}s")
        print(f"    - Other: {other_avg_time:.3f}s")
        print(f"    - Speedup: {other_avg_time/current_avg_time:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deer detection model")
    parser.add_argument(
        "input",
        help="Path to test image, directory, or video"
    )
    parser.add_argument(
        "--model",
        help="Path to model file (default: use deployed model)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save annotated outputs"
    )
    parser.add_argument(
        "--video-sample-rate",
        type=int,
        default=30,
        help="Sample every N frames for video (default: 30)"
    )
    parser.add_argument(
        "--compare-with",
        help="Compare with another model"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model)
    
    # Determine input type and evaluate
    input_path = Path(args.input)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video file
            evaluator.evaluate_video(
                input_path,
                args.confidence,
                args.video_sample_rate
            )
        else:
            # Image file
            evaluator.evaluate_image(
                input_path,
                args.confidence,
                args.save_outputs
            )
    elif input_path.is_dir():
        # Directory of images
        evaluator.evaluate_directory(
            input_path,
            args.confidence,
            args.save_outputs
        )
    else:
        print(f"❌ Invalid input: {input_path}")
        return
    
    # Generate report
    evaluator.generate_report()
    
    # Compare models if requested
    if args.compare_with and input_path.is_dir():
        evaluator.compare_models(args.compare_with, input_path)


if __name__ == "__main__":
    main()