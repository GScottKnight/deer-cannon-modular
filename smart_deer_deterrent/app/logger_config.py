import logging
import logging.handlers
import os
from datetime import datetime

class LoggerConfig:
    """Centralized logging configuration for the Smart Deer Deterrent system."""
    
    def __init__(self, log_dir='logs', log_level=logging.INFO):
        self.log_dir = log_dir
        self.log_level = log_level
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Define log files
        self.log_files = {
            'app': os.path.join(log_dir, 'flask_app.log'),
            'detection': os.path.join(log_dir, 'detections.log'),
            'camera': os.path.join(log_dir, 'camera.log'),
            'system': os.path.join(log_dir, 'system.log'),
            'debug': os.path.join(log_dir, 'debug.log')
        }
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
    def setup_logger(self, name, log_file, level=None):
        """Set up a logger with rotating file handler."""
        logger = logging.getLogger(name)
        logger.setLevel(level or self.log_level)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Create rotating file handler (10MB max, keep 5 backups)
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handler.setFormatter(self.detailed_formatter)
        logger.addHandler(handler)
        
        # Also log to console in debug mode
        if self.log_level == logging.DEBUG:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.simple_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def get_app_logger(self):
        """Get logger for Flask application events."""
        return self.setup_logger('app', self.log_files['app'])
    
    def get_detection_logger(self):
        """Get logger for detection events."""
        logger = self.setup_logger('detection', self.log_files['detection'])
        # Always log detections at INFO level
        logger.setLevel(logging.INFO)
        return logger
    
    def get_camera_logger(self):
        """Get logger for camera-related events."""
        return self.setup_logger('camera', self.log_files['camera'])
    
    def get_system_logger(self):
        """Get logger for system monitoring."""
        return self.setup_logger('system', self.log_files['system'])
    
    def get_debug_logger(self):
        """Get logger for debug information."""
        return self.setup_logger('debug', self.log_files['debug'], logging.DEBUG)
    
    def set_debug_mode(self, enabled):
        """Enable or disable debug mode for all loggers."""
        level = logging.DEBUG if enabled else logging.INFO
        self.log_level = level
        
        # Update all existing loggers
        for logger_name in ['app', 'detection', 'camera', 'system']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            
    def get_recent_logs(self, log_type='app', lines=100):
        """Get recent log entries from a specific log file."""
        log_file = self.log_files.get(log_type, self.log_files['app'])
        
        try:
            with open(log_file, 'r') as f:
                # Read all lines and return the last N lines
                all_lines = f.readlines()
                return all_lines[-lines:]
        except FileNotFoundError:
            return []
        except Exception as e:
            return [f"Error reading log file: {str(e)}"]
    
    def log_detection_event(self, detections, source, video_path=None):
        """Log a detection event with structured information."""
        logger = self.get_detection_logger()
        
        detection_summary = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'detection_count': len(detections),
            'detections': detections,
            'video_saved': video_path is not None,
            'video_path': video_path
        }
        
        # Log as JSON-like string for easy parsing
        logger.info(f"DETECTION_EVENT: {detection_summary}")
        
        # Also log human-readable summary
        animal_types = set(d['label'] for d in detections)
        logger.info(f"Detected {len(detections)} object(s): {', '.join(animal_types)} from {source}")

# Create global logger configuration instance
logger_config = LoggerConfig(log_dir=os.path.join(os.path.dirname(__file__), 'logs'))