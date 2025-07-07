#!/usr/bin/env python3
"""
Azure Blob Storage Uploader for Deer Detection Videos

This module monitors the detection folder and automatically uploads
new detection videos and metadata to Azure Blob Storage.
"""

import os
import sys
import time
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings
from azure.core.exceptions import AzureError
from dotenv import load_dotenv
import cv2

# Load environment variables
load_dotenv()

# Configuration from environment
AZURE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
AZURE_ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
AZURE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'deer-detections')
UPLOAD_BATCH_SIZE = int(os.getenv('UPLOAD_BATCH_SIZE', '5'))
UPLOAD_RETRY_COUNT = int(os.getenv('UPLOAD_RETRY_COUNT', '3'))
UPLOAD_RETRY_DELAY = int(os.getenv('UPLOAD_RETRY_DELAY', '60'))
DELETE_AFTER_UPLOAD = os.getenv('DELETE_AFTER_UPLOAD', 'false').lower() == 'true'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('azure_uploader')

# File paths
SCRIPT_DIR = Path(__file__).parent
DETECTIONS_DIR = SCRIPT_DIR / 'app' / 'detections'
DB_PATH = SCRIPT_DIR / 'upload_queue.db'


class UploadTracker:
    """Track upload status in SQLite database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS uploads (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT,
                    status TEXT CHECK(status IN ('pending', 'uploading', 'completed', 'failed')),
                    attempts INTEGER DEFAULT 0,
                    last_attempt TIMESTAMP,
                    uploaded_at TIMESTAMP,
                    azure_url TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_status ON uploads(status)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_created ON uploads(created_at)
            ''')
    
    def add_file(self, file_path: str, file_hash: str):
        """Add a new file to the upload queue."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR IGNORE INTO uploads (file_path, file_hash, status)
                VALUES (?, ?, 'pending')
            ''', (file_path, file_hash))
    
    def get_pending_files(self, limit: int = None) -> List[Dict]:
        """Get files pending upload."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = '''
                SELECT * FROM uploads 
                WHERE status IN ('pending', 'failed') 
                AND (attempts < ? OR last_attempt < datetime('now', '-1 hour'))
                ORDER BY created_at
            '''
            if limit:
                query += f' LIMIT {limit}'
            
            rows = conn.execute(query, (UPLOAD_RETRY_COUNT,)).fetchall()
            return [dict(row) for row in rows]
    
    def update_status(self, file_path: str, status: str, azure_url: str = None, error: str = None):
        """Update file upload status."""
        with sqlite3.connect(self.db_path) as conn:
            if status == 'completed':
                conn.execute('''
                    UPDATE uploads 
                    SET status = ?, uploaded_at = CURRENT_TIMESTAMP, azure_url = ?
                    WHERE file_path = ?
                ''', (status, azure_url, file_path))
            elif status == 'failed':
                conn.execute('''
                    UPDATE uploads 
                    SET status = ?, attempts = attempts + 1, 
                        last_attempt = CURRENT_TIMESTAMP, error_message = ?
                    WHERE file_path = ?
                ''', (status, error, file_path))
            else:
                conn.execute('''
                    UPDATE uploads 
                    SET status = ?, last_attempt = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                ''', (status, file_path))
    
    def is_uploaded(self, file_path: str) -> bool:
        """Check if file is already uploaded."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('''
                SELECT status FROM uploads WHERE file_path = ? AND status = 'completed'
            ''', (file_path,)).fetchone()
            return result is not None


class AzureUploader:
    """Handle uploads to Azure Blob Storage."""
    
    def __init__(self):
        self.blob_service = self._create_blob_service()
        self.tracker = UploadTracker(DB_PATH)
        
    def _create_blob_service(self) -> BlobServiceClient:
        """Create Azure Blob Service client."""
        if AZURE_CONNECTION_STRING:
            return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        elif AZURE_ACCOUNT_NAME and AZURE_ACCOUNT_KEY:
            return BlobServiceClient(
                account_url=f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=AZURE_ACCOUNT_KEY
            )
        else:
            raise ValueError("Azure credentials not configured. Check your .env file.")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def generate_thumbnail(self, video_path: str, output_path: str) -> bool:
        """Generate thumbnail from first frame of video."""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Resize to thumbnail size (max 320px wide)
                height, width = frame.shape[:2]
                if width > 320:
                    scale = 320 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                cv2.imwrite(output_path, frame)
                return True
        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}")
        return False
    
    def upload_file(self, file_path: Path, blob_name: str) -> Optional[str]:
        """Upload a file to Azure Blob Storage."""
        try:
            # Determine content type
            content_type = 'video/mp4' if file_path.suffix == '.mp4' else 'application/json'
            
            blob_client = self.blob_service.get_blob_client(
                container=AZURE_CONTAINER_NAME, 
                blob=blob_name
            )
            
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(
                    data, 
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type)
                )
            
            return blob_client.url
        except AzureError as e:
            logger.error(f"Azure upload error for {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading {file_path}: {e}")
            raise
    
    def process_detection(self, video_path: Path) -> bool:
        """Process and upload a detection video with metadata."""
        try:
            # Check if already uploaded
            if self.tracker.is_uploaded(str(video_path)):
                logger.info(f"Already uploaded: {video_path}")
                return True
            
            # Update status to uploading
            self.tracker.update_status(str(video_path), 'uploading')
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(str(video_path))
            
            # Derive paths
            json_path = video_path.with_suffix('.json')
            date_folder = video_path.parent.name
            base_name = video_path.stem
            
            # Generate thumbnail
            thumb_path = video_path.with_suffix('.jpg')
            self.generate_thumbnail(str(video_path), str(thumb_path))
            
            # Upload video
            video_blob_name = f"videos/{date_folder}/{video_path.name}"
            video_url = self.upload_file(video_path, video_blob_name)
            logger.info(f"Uploaded video: {video_blob_name}")
            
            # Upload metadata if exists
            if json_path.exists():
                json_blob_name = f"videos/{date_folder}/{json_path.name}"
                self.upload_file(json_path, json_blob_name)
                logger.info(f"Uploaded metadata: {json_blob_name}")
            
            # Upload thumbnail if generated
            if thumb_path.exists():
                thumb_blob_name = f"thumbnails/{date_folder}/{thumb_path.name}"
                self.upload_file(thumb_path, thumb_blob_name)
                logger.info(f"Uploaded thumbnail: {thumb_blob_name}")
                # Clean up local thumbnail
                thumb_path.unlink()
            
            # Update tracker
            self.tracker.update_status(str(video_path), 'completed', video_url)
            
            # Delete local files if configured
            if DELETE_AFTER_UPLOAD:
                video_path.unlink()
                if json_path.exists():
                    json_path.unlink()
                logger.info(f"Deleted local files for {base_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            self.tracker.update_status(str(video_path), 'failed', error=str(e))
            return False
    
    def scan_for_new_files(self):
        """Scan detection directory for new files."""
        logger.info("Scanning for new detection files...")
        
        for date_dir in DETECTIONS_DIR.iterdir():
            if not date_dir.is_dir():
                continue
                
            for video_file in date_dir.glob("detection_*.mp4"):
                if not self.tracker.is_uploaded(str(video_file)):
                    file_hash = self._calculate_file_hash(str(video_file))
                    self.tracker.add_file(str(video_file), file_hash)
                    logger.info(f"Added to queue: {video_file}")
    
    def process_queue(self):
        """Process pending uploads from queue."""
        pending = self.tracker.get_pending_files(limit=UPLOAD_BATCH_SIZE)
        
        if not pending:
            logger.debug("No pending uploads")
            return
        
        logger.info(f"Processing {len(pending)} pending uploads")
        
        for file_info in pending:
            file_path = Path(file_info['file_path'])
            
            if not file_path.exists():
                logger.warning(f"File no longer exists: {file_path}")
                self.tracker.update_status(str(file_path), 'failed', error='File not found')
                continue
            
            success = self.process_detection(file_path)
            if success:
                logger.info(f"Successfully uploaded: {file_path.name}")
            else:
                logger.warning(f"Failed to upload: {file_path.name}")
    
    def update_web_index(self):
        """Update the detection index for the web interface."""
        try:
            # Generate updated index
            from generate_web_index import generate_index
            generate_index()
            
            # Upload to Azure
            index_path = SCRIPT_DIR / 'public_website' / 'api' / 'detections.json'
            if index_path.exists():
                blob_client = self.blob_service.get_blob_client(
                    container=AZURE_CONTAINER_NAME,
                    blob='api/detections.json'
                )
                with open(index_path, 'rb') as f:
                    blob_client.upload_blob(
                        f, 
                        overwrite=True,
                        content_settings=ContentSettings(content_type='application/json')
                    )
                logger.info("Updated web index")
        except Exception as e:
            logger.error(f"Failed to update web index: {e}")
    
    def run(self, scan_interval: int = 60):
        """Run the upload service continuously."""
        logger.info("Starting Azure upload service...")
        logger.info(f"Monitoring: {DETECTIONS_DIR}")
        logger.info(f"Container: {AZURE_CONTAINER_NAME}")
        
        while True:
            try:
                # Scan for new files
                self.scan_for_new_files()
                
                # Process upload queue
                self.process_queue()
                
                # Update web index
                self.update_web_index()
                
                # Wait before next scan
                logger.debug(f"Sleeping for {scan_interval} seconds...")
                time.sleep(scan_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutting down upload service")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(scan_interval)


def main():
    """Main entry point."""
    # Validate configuration
    if not AZURE_ACCOUNT_NAME or not AZURE_ACCOUNT_KEY:
        if not AZURE_CONNECTION_STRING:
            logger.error("Azure credentials not configured. Copy .env.template to .env and add your credentials.")
            sys.exit(1)
    
    # Create container if needed
    try:
        uploader = AzureUploader()
        container_client = uploader.blob_service.get_container_client(AZURE_CONTAINER_NAME)
        if not container_client.exists():
            container_client.create_container(public_access='blob')
            logger.info(f"Created container: {AZURE_CONTAINER_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)
    
    # Run the service
    uploader.run()


if __name__ == '__main__':
    main()