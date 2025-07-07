#!/usr/bin/env python3
"""
Download YOLO models from Azure Blob Storage
Uses existing Azure credentials from .env file
"""

import os
import sys
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure configuration
AZURE_STORAGE_ACCOUNT = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
AZURE_STORAGE_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
MODELS_CONTAINER = 'models'

def create_blob_service_client():
    """Create Azure Blob Service client."""
    if not AZURE_STORAGE_ACCOUNT or not AZURE_STORAGE_KEY:
        print("Error: Azure credentials not found in .env file")
        print("Please ensure .env file exists with AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY")
        sys.exit(1)
    
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={AZURE_STORAGE_ACCOUNT};AccountKey={AZURE_STORAGE_KEY};EndpointSuffix=core.windows.net"
    return BlobServiceClient.from_connection_string(connection_string)

def list_available_models(blob_service_client):
    """List all available models in Azure."""
    try:
        container_client = blob_service_client.get_container_client(MODELS_CONTAINER)
        blobs = list(container_client.list_blobs())
        
        if not blobs:
            print("No models found in Azure storage.")
            return []
        
        print("\n=== Available Models in Azure ===")
        models = []
        for i, blob in enumerate(blobs):
            print(f"{i+1}. {blob.name} ({blob.size / 1024 / 1024:.1f} MB)")
            models.append(blob)
        
        return models
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def download_model(blob_service_client, blob_name, local_path):
    """Download a model from Azure."""
    try:
        blob_client = blob_service_client.get_blob_client(
            container=MODELS_CONTAINER,
            blob=blob_name
        )
        
        # Create directory if needed
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {blob_name}...")
        
        with open(local_path, 'wb') as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())
        
        print(f"✓ Downloaded to: {local_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {blob_name}: {e}")
        return False

def download_latest_models():
    """Download the latest version of each model."""
    print("=== Download YOLO Models from Azure ===\n")
    
    # Create blob service client
    blob_service_client = create_blob_service_client()
    
    # List available models
    models = list_available_models(blob_service_client)
    if not models:
        return
    
    # Group models by filename
    model_groups = {}
    for blob in models:
        # Extract filename from path (e.g., "20240101/app/models/best.pt" -> "best.pt")
        filename = Path(blob.name).name
        if filename not in model_groups:
            model_groups[filename] = []
        model_groups[filename].append(blob)
    
    print("\n=== Downloading Latest Models ===\n")
    
    for filename, blobs in model_groups.items():
        # Sort by name (which includes timestamp) to get latest
        latest_blob = sorted(blobs, key=lambda b: b.name)[-1]
        
        # Determine local path
        if 'app/models' in latest_blob.name:
            local_path = f"app/models/{filename}"
        elif 'camera_detection/models' in latest_blob.name:
            local_path = f"camera_detection/models/{filename}"
        else:
            local_path = f"app/models/{filename}"  # Default location
        
        download_model(blob_service_client, latest_blob.name, local_path)
    
    print("\nDownload complete!")

def download_specific_model():
    """Interactive download of specific model."""
    print("=== Download Specific Model from Azure ===\n")
    
    # Create blob service client
    blob_service_client = create_blob_service_client()
    
    # List available models
    models = list_available_models(blob_service_client)
    if not models:
        return
    
    # Get user selection
    try:
        choice = input("\nEnter model number to download (or 'all' for latest versions): ").strip()
        
        if choice.lower() == 'all':
            download_latest_models()
            return
        
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            blob = models[idx]
            
            # Ask for local path
            default_path = Path(blob.name).name
            if 'app/models' in blob.name:
                default_path = f"app/models/{default_path}"
            elif 'camera_detection/models' in blob.name:
                default_path = f"camera_detection/models/{default_path}"
            
            local_path = input(f"Local path [{default_path}]: ").strip() or default_path
            
            download_model(blob_service_client, blob.name, local_path)
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--latest':
        download_latest_models()
    else:
        download_specific_model()