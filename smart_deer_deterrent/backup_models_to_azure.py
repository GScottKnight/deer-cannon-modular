#!/usr/bin/env python3
"""
Backup YOLO models to Azure Blob Storage
Uses existing Azure credentials from .env file
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure configuration
AZURE_STORAGE_ACCOUNT = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
AZURE_STORAGE_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
MODELS_CONTAINER = 'models'  # Container for model backups

def create_blob_service_client():
    """Create Azure Blob Service client."""
    if not AZURE_STORAGE_ACCOUNT or not AZURE_STORAGE_KEY:
        print("Error: Azure credentials not found in .env file")
        sys.exit(1)
    
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={AZURE_STORAGE_ACCOUNT};AccountKey={AZURE_STORAGE_KEY};EndpointSuffix=core.windows.net"
    return BlobServiceClient.from_connection_string(connection_string)

def ensure_container_exists(blob_service_client, container_name):
    """Create container if it doesn't exist."""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container(public_access='blob')
            print(f"Created container: {container_name}")
        else:
            print(f"Container exists: {container_name}")
    except Exception as e:
        print(f"Error creating container: {e}")
        sys.exit(1)

def upload_model(blob_service_client, local_path, blob_name):
    """Upload a model file to Azure."""
    try:
        blob_client = blob_service_client.get_blob_client(
            container=MODELS_CONTAINER, 
            blob=blob_name
        )
        
        file_size = os.path.getsize(local_path)
        print(f"Uploading {local_path} ({file_size / 1024 / 1024:.1f} MB)...")
        
        with open(local_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        print(f"✓ Uploaded to: {blob_name}")
        
        # Generate download URL
        url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{MODELS_CONTAINER}/{blob_name}"
        print(f"  Download URL: {url}")
        
        return url
    except Exception as e:
        print(f"✗ Error uploading {local_path}: {e}")
        return None

def backup_models():
    """Backup all model files to Azure."""
    print("=== YOLO Model Backup to Azure ===\n")
    
    # Create blob service client
    blob_service_client = create_blob_service_client()
    
    # Ensure container exists
    ensure_container_exists(blob_service_client, MODELS_CONTAINER)
    
    # Find model files
    model_dirs = [
        'app/models',
        'camera_detection/models'
    ]
    
    model_files = []
    for model_dir in model_dirs:
        model_path = Path(model_dir)
        if model_path.exists():
            model_files.extend(model_path.glob('*.pt'))
            model_files.extend(model_path.glob('*.pth'))
    
    if not model_files:
        print("No model files found!")
        return
    
    print(f"\nFound {len(model_files)} model files to backup:\n")
    
    # Upload each model
    uploaded_urls = {}
    timestamp = datetime.now().strftime('%Y%m%d')
    
    for model_file in model_files:
        # Create blob name with timestamp and original structure
        relative_path = model_file.relative_to('.')
        blob_name = f"{timestamp}/{relative_path}"
        
        url = upload_model(blob_service_client, model_file, blob_name)
        if url:
            uploaded_urls[str(relative_path)] = url
    
    # Save download URLs to a file
    if uploaded_urls:
        urls_file = Path('model_download_urls.txt')
        with open(urls_file, 'w') as f:
            f.write(f"Model Backup URLs - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            for path, url in uploaded_urls.items():
                f.write(f"{path}:\n{url}\n\n")
        
        print(f"\n✓ Download URLs saved to: {urls_file}")
        print("\nBackup complete!")
    
    # Also list all models in container
    print(f"\n=== All Models in Azure Container '{MODELS_CONTAINER}' ===")
    container_client = blob_service_client.get_container_client(MODELS_CONTAINER)
    blobs = list(container_client.list_blobs())
    for blob in blobs[-10:]:  # Show last 10
        print(f"  {blob.name} ({blob.size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    backup_models()