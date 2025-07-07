#!/usr/bin/env python3
"""
Generate detection index JSON for the web interface.

This script creates a JSON index of all uploaded detections
that the web interface can load to display the gallery.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure configuration
AZURE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
AZURE_ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
AZURE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'deer-detections')
WEB_INTERFACE_URL = os.getenv('WEB_INTERFACE_URL', '')

# Output path
OUTPUT_DIR = Path(__file__).parent / 'public_website' / 'api'
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / 'detections.json'


def get_blob_service() -> BlobServiceClient:
    """Create Azure Blob Service client."""
    if AZURE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    elif AZURE_ACCOUNT_NAME and AZURE_ACCOUNT_KEY:
        return BlobServiceClient(
            account_url=f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=AZURE_ACCOUNT_KEY
        )
    else:
        raise ValueError("Azure credentials not configured.")


def parse_detection_filename(filename: str) -> Dict:
    """Parse detection filename to extract timestamp."""
    # Format: detection_YYYYMMDD_HHMMSS.mp4
    try:
        parts = filename.replace('detection_', '').replace('.mp4', '').split('_')
        date_str = parts[0]
        time_str = parts[1]
        
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
        timestamp = f"{date}T{time}"
        
        return {
            'date': date,
            'time': time,
            'timestamp': timestamp
        }
    except:
        return None


def generate_index():
    """Generate detection index from Azure Blob Storage."""
    print("Connecting to Azure Blob Storage...")
    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(AZURE_CONTAINER_NAME)
    
    detections = []
    video_blobs = {}
    json_blobs = {}
    thumbnail_blobs = {}
    
    # List all blobs
    print("Listing blobs...")
    for blob in container_client.list_blobs():
        if blob.name.startswith('videos/') and blob.name.endswith('.mp4'):
            key = blob.name.replace('videos/', '').replace('.mp4', '')
            video_blobs[key] = blob
        elif blob.name.startswith('videos/') and blob.name.endswith('.json'):
            key = blob.name.replace('videos/', '').replace('.json', '')
            json_blobs[key] = blob
        elif blob.name.startswith('thumbnails/') and blob.name.endswith('.jpg'):
            key = blob.name.replace('thumbnails/', '').replace('.jpg', '')
            thumbnail_blobs[key] = blob
    
    print(f"Found {len(video_blobs)} videos")
    
    # Process each video
    for key, video_blob in video_blobs.items():
        filename = os.path.basename(video_blob.name)
        parsed = parse_detection_filename(filename)
        
        if not parsed:
            continue
        
        detection = {
            'id': key.split('/')[-1],
            'timestamp': parsed['timestamp'],
            'date': parsed['date'],
            'time': parsed['time'],
            'video_url': f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{video_blob.name}",
            'video_size': video_blob.size,
            'uploaded_at': video_blob.last_modified.isoformat() if video_blob.last_modified else None
        }
        
        # Add thumbnail URL if exists
        if key in thumbnail_blobs:
            detection['thumbnail_url'] = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{thumbnail_blobs[key].name}"
        
        # Load metadata if exists
        if key in json_blobs:
            try:
                blob_client = blob_service.get_blob_client(
                    container=AZURE_CONTAINER_NAME,
                    blob=json_blobs[key].name
                )
                metadata_str = blob_client.download_blob().readall()
                metadata = json.loads(metadata_str)
                
                detection['confidence'] = metadata.get('detections', [{}])[0].get('confidence', 0.5)
                detection['labels'] = list(set(d.get('label', 'unknown') for d in metadata.get('detections', [])))
                detection['metadata'] = {
                    'duration': metadata.get('source', {}).get('duration_frames', 0) / 10,  # Assuming 10 fps
                    'fps': metadata.get('source', {}).get('fps', 10),
                    'detection_count': metadata.get('detection_count', 0)
                }
            except Exception as e:
                print(f"Error loading metadata for {key}: {e}")
                detection['confidence'] = 0.5
                detection['labels'] = ['unknown']
                detection['metadata'] = {}
        else:
            detection['confidence'] = 0.5
            detection['labels'] = ['unknown']
            detection['metadata'] = {}
        
        detections.append(detection)
    
    # Sort by timestamp (newest first)
    detections.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Create index
    index = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'total_count': len(detections),
        'storage_account': AZURE_ACCOUNT_NAME,
        'container': AZURE_CONTAINER_NAME,
        'detections': detections
    }
    
    # Save to file
    print(f"Writing index to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Generated index with {len(detections)} detections")
    
    # Also upload to Azure for web access
    try:
        blob_client = blob_service.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob='api/detections.json'
        )
        with open(OUTPUT_FILE, 'rb') as f:
            blob_client.upload_blob(f, overwrite=True)
        print("Index uploaded to Azure")
    except Exception as e:
        print(f"Warning: Could not upload index to Azure: {e}")


def update_js_config():
    """Update JavaScript configuration with Azure account name."""
    js_file = Path(__file__).parent / 'public_website' / 'js' / 'app.js'
    
    if js_file.exists():
        content = js_file.read_text()
        content = content.replace(
            'YOUR_ACCOUNT',
            AZURE_ACCOUNT_NAME
        )
        js_file.write_text(content)
        print(f"Updated JavaScript configuration with account: {AZURE_ACCOUNT_NAME}")


if __name__ == '__main__':
    if not AZURE_ACCOUNT_NAME:
        print("Error: Azure credentials not configured. Set up your .env file first.")
        exit(1)
    
    generate_index()
    update_js_config()