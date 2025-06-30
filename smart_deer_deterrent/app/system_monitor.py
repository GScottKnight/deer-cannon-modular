import psutil
import os
import json
from datetime import datetime, timedelta
import platform

class SystemMonitor:
    """Monitor system resources and application health."""
    
    def __init__(self, app_start_time=None):
        self.start_time = app_start_time or datetime.now()
        
    def get_system_info(self):
        """Get comprehensive system information."""
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory info
        memory = psutil.virtual_memory()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        # Network info (if available)
        try:
            net_io = psutil.net_io_counters()
            network_info = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
        except:
            network_info = {'status': 'unavailable'}
        
        # Temperature (if available)
        temps = {}
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                sensors = psutil.sensors_temperatures()
                for name, entries in sensors.items():
                    for entry in entries:
                        temps[f"{name}_{entry.label}"] = entry.current
        except:
            temps = {'status': 'unavailable'}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - self.start_time),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'python_version': platform.python_version()
            },
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency': psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else 'N/A'
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'network': network_info,
            'temperatures': temps
        }
    
    def get_process_info(self):
        """Get information about the current process."""
        process = psutil.Process(os.getpid())
        
        return {
            'pid': process.pid,
            'cpu_percent': process.cpu_percent(),
            'memory_info': process.memory_info()._asdict(),
            'num_threads': process.num_threads(),
            'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
        }
    
    def check_disk_space(self, min_free_gb=1):
        """Check if disk space is sufficient."""
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        
        return {
            'free_gb': round(free_gb, 2),
            'is_sufficient': free_gb >= min_free_gb,
            'min_required_gb': min_free_gb
        }
    
    def get_detection_stats(self, detection_folder):
        """Get statistics about detection videos."""
        stats = {
            'total_detections': 0,
            'total_size_mb': 0,
            'detections_by_date': {},
            'latest_detection': None
        }
        
        if not os.path.exists(detection_folder):
            return stats
        
        # Walk through detection directories
        for date_dir in os.listdir(detection_folder):
            date_path = os.path.join(detection_folder, date_dir)
            if not os.path.isdir(date_path):
                continue
                
            date_count = 0
            date_size = 0
            
            for filename in os.listdir(date_path):
                if filename.endswith('.mp4'):
                    stats['total_detections'] += 1
                    date_count += 1
                    
                    file_path = os.path.join(date_path, filename)
                    file_size = os.path.getsize(file_path) / (1024**2)  # MB
                    stats['total_size_mb'] += file_size
                    date_size += file_size
                    
                    # Track latest detection
                    file_time = os.path.getmtime(file_path)
                    if not stats['latest_detection'] or file_time > stats['latest_detection']['timestamp']:
                        stats['latest_detection'] = {
                            'filename': filename,
                            'timestamp': file_time,
                            'datetime': datetime.fromtimestamp(file_time).isoformat(),
                            'size_mb': round(file_size, 2)
                        }
            
            if date_count > 0:
                stats['detections_by_date'][date_dir] = {
                    'count': date_count,
                    'size_mb': round(date_size, 2)
                }
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats

# Create global monitor instance
system_monitor = SystemMonitor()