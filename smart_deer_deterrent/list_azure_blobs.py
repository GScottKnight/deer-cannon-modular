#!/usr/bin/env python3
"""
List all blobs in Azure storage to verify paths.
"""

import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure configuration
AZURE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'deer-detections')

def list_blobs():
    """List all blobs in the container."""
    print("Connecting to Azure Storage...")
    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(AZURE_CONTAINER_NAME)
    
    print(f"\nListing all blobs in container '{AZURE_CONTAINER_NAME}':\n")
    
    blobs = list(container_client.list_blobs())
    
    # Group by type
    videos = []
    thumbnails = []
    other = []
    
    for blob in blobs:
        if blob.name.startswith('videos/'):
            videos.append(blob.name)
        elif blob.name.startswith('thumbnails/'):
            thumbnails.append(blob.name)
        else:
            other.append(blob.name)
    
    print(f"Videos ({len(videos)}):")
    for v in sorted(videos):
        print(f"  {v}")
    
    print(f"\nThumbnails ({len(thumbnails)}):")
    for t in sorted(thumbnails):
        print(f"  {t}")
        
    print(f"\nOther files ({len(other)}):")
    for o in sorted(other):
        print(f"  {o}")
    
    # Check specific video
    print("\nChecking for specific video:")
    target = "videos/2025-06-30/detection_20250630_180205.mp4"
    exists = any(blob.name == target for blob in blobs)
    print(f"  {target}: {'EXISTS' if exists else 'NOT FOUND'}")

if __name__ == '__main__':
    list_blobs()