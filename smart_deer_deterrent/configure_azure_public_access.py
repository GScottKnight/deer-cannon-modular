#!/usr/bin/env python3
"""
Configure Azure Blob Storage for public access and CORS.
"""

import os
from azure.storage.blob import BlobServiceClient, PublicAccess, CorsRule
from azure.core.exceptions import ResourceExistsError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure configuration
AZURE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'deer-detections')

def configure_public_access():
    """Configure container for public access and CORS."""
    print("Connecting to Azure Storage...")
    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    
    # Get container client
    container_client = blob_service.get_container_client(AZURE_CONTAINER_NAME)
    
    # Set public access level
    print(f"Setting public access on container '{AZURE_CONTAINER_NAME}'...")
    try:
        container_client.set_container_access_policy(
            signed_identifiers={},
            public_access=PublicAccess.Blob
        )
        print("✓ Public access enabled (anonymous read access for blobs)")
    except Exception as e:
        print(f"Error setting public access: {e}")
    
    # Configure CORS
    print("\nConfiguring CORS rules...")
    cors_rule = CorsRule(
        allowed_origins=['*'],  # Allow all origins
        allowed_methods=['GET', 'HEAD', 'OPTIONS', 'PUT'],
        allowed_headers=['*'],
        exposed_headers=['*', 'Content-Length', 'Content-Type', 'Content-Range', 'Accept-Ranges'],
        max_age_in_seconds=3600
    )
    
    try:
        blob_service.set_service_properties(
            cors=[cors_rule]
        )
        print("✓ CORS rules configured")
    except Exception as e:
        print(f"Error setting CORS: {e}")
    
    # Set proper content types for HTML files
    print("\nUpdating content types for web files...")
    web_files = {
        'index.html': 'text/html',
        'css/style.css': 'text/css',
        'js/app.js': 'application/javascript',
        'api/detections.json': 'application/json'
    }
    
    for blob_name, content_type in web_files.items():
        try:
            blob_client = blob_service.get_blob_client(
                container=AZURE_CONTAINER_NAME,
                blob=blob_name
            )
            from azure.storage.blob import ContentSettings
            blob_client.set_http_headers(content_settings=ContentSettings(content_type=content_type))
            print(f"✓ Set content type for {blob_name}")
        except Exception as e:
            print(f"  Warning: Could not update {blob_name}: {e}")
    
    print("\n✅ Configuration complete!")
    print(f"\nYour detection gallery is now publicly accessible at:")
    print(f"https://deerdetections.blob.core.windows.net/{AZURE_CONTAINER_NAME}/index.html")
    print("\nDirect video links are also accessible:")
    print("https://deerdetections.blob.core.windows.net/deer-detections/videos/2025-06-30/detection_20250630_180205.mp4")

if __name__ == '__main__':
    configure_public_access()