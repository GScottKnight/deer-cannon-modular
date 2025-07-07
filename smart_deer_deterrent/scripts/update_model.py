#!/usr/bin/env python3
"""
Model Update Script for Smart Deer Deterrent System

This script helps transfer trained models from the training project
to the deployment project with proper validation and versioning.
"""

import os
import shutil
import hashlib
import json
import argparse
from datetime import datetime
from pathlib import Path


class ModelUpdater:
    def __init__(self, training_project_path, deployment_project_path):
        self.training_path = Path(training_project_path)
        self.deployment_path = Path(deployment_project_path)
        self.models_dir = self.deployment_path / "camera_detection" / "models"
        self.versions_file = self.models_dir / "model_versions.json"
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load version history
        self.version_history = self._load_version_history()
    
    def _load_version_history(self):
        """Load model version history from JSON file."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {"versions": [], "active": None}
    
    def _save_version_history(self):
        """Save model version history to JSON file."""
        with open(self.versions_file, 'w') as f:
            json.dump(self.version_history, f, indent=2)
    
    def _calculate_checksum(self, file_path):
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _find_best_model(self):
        """Find the best model in the training project."""
        # Look for model in common locations
        possible_paths = [
            self.training_path / "runs" / "detect" / "train" / "weights" / "best.pt",
            self.training_path / "deer_detection" / "deer_detection_run" / "weights" / "best.pt",
            self.training_path / "deer_detection" / "deer_detection_run2" / "weights" / "best.pt",
            self.training_path / "best.pt",
        ]
        
        # Find the most recent model
        best_model = None
        latest_time = 0
        
        for path in possible_paths:
            if path.exists():
                mod_time = path.stat().st_mtime
                if mod_time > latest_time:
                    latest_time = mod_time
                    best_model = path
        
        return best_model
    
    def _get_model_info(self, model_path):
        """Extract model information."""
        # Try to read training results if available
        results_path = model_path.parent.parent / "results.csv"
        metrics = {}
        
        if results_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(results_path)
                if not df.empty:
                    last_row = df.iloc[-1]
                    metrics = {
                        "mAP50": float(last_row.get("metrics/mAP50(B)", 0)),
                        "mAP50-95": float(last_row.get("metrics/mAP50-95(B)", 0)),
                        "precision": float(last_row.get("metrics/precision(B)", 0)),
                        "recall": float(last_row.get("metrics/recall(B)", 0)),
                    }
            except:
                pass
        
        return {
            "source_path": str(model_path),
            "file_size": model_path.stat().st_size,
            "checksum": self._calculate_checksum(model_path),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
    
    def update_model(self, model_name="deer", force=False):
        """Update a model from training to deployment project."""
        # Find the best model
        source_model = self._find_best_model()
        if not source_model:
            print("❌ No trained model found in the training project!")
            return False
        
        print(f"📁 Found model: {source_model}")
        
        # Get model info
        model_info = self._get_model_info(source_model)
        
        # Check if this model is already deployed
        for version in self.version_history["versions"]:
            if version["checksum"] == model_info["checksum"] and not force:
                print("✅ This model is already deployed!")
                print(f"   Version: {version['version']}")
                return True
        
        # Create version number
        version_num = len(self.version_history["versions"]) + 1
        version_name = f"v{version_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Copy model with versioned name
        versioned_path = self.models_dir / f"{model_name}_model_{version_name}.pt"
        shutil.copy2(source_model, versioned_path)
        print(f"📦 Copied model to: {versioned_path}")
        
        # Also update the main model file
        main_model_path = self.models_dir / "best.pt"
        shutil.copy2(source_model, main_model_path)
        print(f"📦 Updated main model: {main_model_path}")
        
        # Update version history
        version_entry = {
            "version": version_name,
            "model_name": model_name,
            "file": versioned_path.name,
            **model_info
        }
        
        self.version_history["versions"].append(version_entry)
        self.version_history["active"] = version_name
        self._save_version_history()
        
        print(f"✅ Model updated successfully!")
        print(f"   Version: {version_name}")
        if model_info["metrics"]:
            print(f"   Metrics:")
            for key, value in model_info["metrics"].items():
                print(f"     - {key}: {value:.3f}")
        
        return True
    
    def list_versions(self):
        """List all model versions."""
        if not self.version_history["versions"]:
            print("No model versions found.")
            return
        
        print("\n📋 Model Version History:")
        print("-" * 80)
        
        for version in reversed(self.version_history["versions"]):
            active = " (ACTIVE)" if version["version"] == self.version_history["active"] else ""
            print(f"\n🔸 {version['version']}{active}")
            print(f"   File: {version['file']}")
            print(f"   Date: {version['timestamp']}")
            print(f"   Size: {version['file_size'] / 1024 / 1024:.1f} MB")
            
            if version.get("metrics"):
                print("   Metrics:")
                for key, value in version["metrics"].items():
                    print(f"     - {key}: {value:.3f}")
    
    def rollback(self, version_name):
        """Rollback to a specific model version."""
        # Find the version
        version_entry = None
        for version in self.version_history["versions"]:
            if version["version"] == version_name:
                version_entry = version
                break
        
        if not version_entry:
            print(f"❌ Version {version_name} not found!")
            return False
        
        # Copy the versioned model to the main model file
        versioned_path = self.models_dir / version_entry["file"]
        if not versioned_path.exists():
            print(f"❌ Model file not found: {versioned_path}")
            return False
        
        main_model_path = self.models_dir / "best.pt"
        shutil.copy2(versioned_path, main_model_path)
        
        # Update active version
        self.version_history["active"] = version_name
        self._save_version_history()
        
        print(f"✅ Rolled back to version: {version_name}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Update deer detection models")
    parser.add_argument(
        "--training-path",
        default="/Users/gsknight/Documents/deer detection training",
        help="Path to training project"
    )
    parser.add_argument(
        "--deployment-path",
        default="/Users/gsknight/Documents/Deer_Cannon_Modular/smart_deer_deterrent",
        help="Path to deployment project"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all model versions"
    )
    parser.add_argument(
        "--rollback",
        type=str,
        help="Rollback to a specific version"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if model already exists"
    )
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = ModelUpdater(args.training_path, args.deployment_path)
    
    # Execute requested action
    if args.list:
        updater.list_versions()
    elif args.rollback:
        updater.rollback(args.rollback)
    else:
        updater.update_model(force=args.force)


if __name__ == "__main__":
    main()