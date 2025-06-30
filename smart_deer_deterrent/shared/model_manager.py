import os
from ultralytics import YOLO

class ModelManager:
    """Singleton class to manage YOLO models and avoid duplicate loading."""
    _instance = None
    _deer_model = None
    _general_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_deer_model(self, model_path=None):
        """Get or load the deer detection model."""
        if self._deer_model is None and model_path:
            self._deer_model = YOLO(model_path)
        return self._deer_model
    
    def get_general_model(self, model_path=None):
        """Get or load the general detection model."""
        if self._general_model is None and model_path:
            self._general_model = YOLO(model_path)
        return self._general_model
    
    def reload_models(self, deer_path=None, general_path=None):
        """Force reload of models (useful for updating models)."""
        if deer_path:
            self._deer_model = YOLO(deer_path)
        if general_path:
            self._general_model = YOLO(general_path)

# Global model manager instance
model_manager = ModelManager()