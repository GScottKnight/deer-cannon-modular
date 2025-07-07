"""
Model Configuration Loader

Loads model configuration and provides easy access to model settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ModelConfig:
    """Singleton class to manage model configuration."""
    
    _instance = None
    _config = None
    _config_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from JSON file."""
        # Find config file
        current_dir = Path(__file__).parent.parent
        self._config_path = current_dir / "config" / "model_config.json"
        
        if self._config_path.exists():
            with open(self._config_path, 'r') as f:
                self._config = json.load(f)
        else:
            # Default configuration if file doesn't exist
            self._config = {
                "models": {
                    "deer": {
                        "path": "camera_detection/models/best.pt",
                        "confidence_threshold": 0.3
                    },
                    "general": {
                        "path": "camera_detection/models/yolov8n.pt",
                        "confidence_threshold": 0.25
                    }
                },
                "inference_settings": {
                    "device": "auto",
                    "use_dual_model": True,
                    "safety_mode": True
                }
            }
    
    def save_config(self):
        """Save current configuration to file."""
        if self._config_path:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self._config.get("models", {}).get(model_name, {})
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the file path for a model."""
        model_config = self.get_model_config(model_name)
        if model_config and "path" in model_config:
            # Convert relative path to absolute
            base_dir = Path(__file__).parent.parent
            return str(base_dir / model_config["path"])
        return None
    
    def get_confidence_threshold(self, model_name: str) -> float:
        """Get confidence threshold for a model."""
        model_config = self.get_model_config(model_name)
        return model_config.get("confidence_threshold", 0.25)
    
    def get_inference_settings(self) -> Dict[str, Any]:
        """Get general inference settings."""
        return self._config.get("inference_settings", {})
    
    def update_model_version(self, model_name: str, version: str, metrics: Optional[Dict] = None):
        """Update model version information."""
        if model_name in self._config["models"]:
            self._config["models"][model_name]["current_version"] = version
            if metrics:
                self._config["models"][model_name]["metrics"] = metrics
            self.save_config()
    
    def get_device(self) -> str:
        """Get device configuration."""
        return self._config.get("inference_settings", {}).get("device", "auto")
    
    def is_dual_model_enabled(self) -> bool:
        """Check if dual model inference is enabled."""
        return self._config.get("inference_settings", {}).get("use_dual_model", True)
    
    def is_safety_mode_enabled(self) -> bool:
        """Check if safety mode is enabled (person detection = animals are pets)."""
        return self._config.get("inference_settings", {}).get("safety_mode", True)


# Global configuration instance
model_config = ModelConfig()