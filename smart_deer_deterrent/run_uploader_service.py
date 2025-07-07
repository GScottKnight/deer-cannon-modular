#!/usr/bin/env python3
"""
Run the Azure uploader service in watch mode.
This will only upload new detections as they are created.
"""

import time
import logging
from pathlib import Path
from azure_uploader import AzureUploader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the uploader service for new detections only."""
    uploader = AzureUploader()
    
    logger.info("Starting Azure uploader service (new detections only)...")
    logger.info("Press Ctrl+C to stop")
    
    # Mark all existing files as already processed
    # This prevents uploading old files
    existing_files = list(Path(uploader.detections_dir).rglob("*.mp4"))
    logger.info(f"Marking {len(existing_files)} existing files as processed...")
    
    for file_path in existing_files:
        # Check if already in database
        uploader.cursor.execute(
            "SELECT status FROM uploads WHERE file_path = ?",
            (str(file_path),)
        )
        result = uploader.cursor.fetchone()
        
        if not result:
            # Add to database as 'skipped' to prevent upload
            uploader.cursor.execute("""
                INSERT INTO uploads (file_path, filename, status, created_at)
                VALUES (?, ?, 'skipped', datetime('now'))
            """, (str(file_path), file_path.name))
    
    uploader.conn.commit()
    logger.info("Ready to upload new detections as they are created")
    
    # Run the service
    try:
        while True:
            # Process any pending uploads (new files only)
            uploader.process_pending_uploads()
            
            # Wait before checking again
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        logger.info("Stopping uploader service...")
        uploader.cleanup()

if __name__ == "__main__":
    main()