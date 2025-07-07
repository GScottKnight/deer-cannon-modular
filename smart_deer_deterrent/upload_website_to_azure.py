#!/usr/bin/env python3
"""
Upload the web interface files to Azure Blob Storage.
This allows the website to be served directly from Azure.
"""

import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure configuration
AZURE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'deer-detections')

# MIME type mappings
CONTENT_TYPES = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.jpg': 'image/jpeg',
    '.png': 'image/png',
    '.ico': 'image/x-icon'
}

def upload_website():
    """Upload all website files to Azure."""
    print("Connecting to Azure...")
    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    
    website_dir = Path(__file__).parent / 'public_website'
    
    for file_path in website_dir.rglob('*'):
        if file_path.is_file():
            # Get relative path for blob name
            relative_path = file_path.relative_to(website_dir)
            blob_name = str(relative_path).replace('\\', '/')
            
            # Get content type
            ext = file_path.suffix.lower()
            content_type = CONTENT_TYPES.get(ext, 'application/octet-stream')
            
            print(f"Uploading {blob_name}...")
            
            # Upload with proper content type
            blob_client = blob_service.get_blob_client(
                container=AZURE_CONTAINER_NAME,
                blob=blob_name
            )
            
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type)
                )
    
    print("\nWebsite uploaded successfully!")
    print(f"Access your gallery at:")
    print(f"https://deerdetections.blob.core.windows.net/{AZURE_CONTAINER_NAME}/index.html")
    print("\nNote: You need to enable public access and CORS in Azure Portal for this to work.")

if __name__ == '__main__':
    upload_website()