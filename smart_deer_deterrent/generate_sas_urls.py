#!/usr/bin/env python3
"""
Generate SAS (Shared Access Signature) URLs for accessing videos without public access.
"""

import os
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Azure configuration
AZURE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
AZURE_ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
AZURE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'deer-detections')

def generate_sas_url(blob_name, days_valid=30):
    """Generate a SAS URL for a blob."""
    sas_token = generate_blob_sas(
        account_name=AZURE_ACCOUNT_NAME,
        container_name=AZURE_CONTAINER_NAME,
        blob_name=blob_name,
        account_key=AZURE_ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(days=days_valid)
    )
    
    # Don't double-encode the SAS token
    return f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob_name}?{sas_token}"

def update_detections_with_sas():
    """Update the detections.json file with SAS URLs."""
    # Load current detections
    detections_file = 'public_website/api/detections.json'
    with open(detections_file, 'r') as f:
        data = json.load(f)
    
    print(f"Generating SAS URLs for {len(data['detections'])} detections...")
    
    # Update each detection with SAS URLs
    for detection in data['detections']:
        # Generate SAS URL for video
        video_blob = f"videos/{detection['date']}/{detection['id']}.mp4"
        detection['video_sas_url'] = generate_sas_url(video_blob)
        
        # Generate SAS URL for thumbnail
        thumbnail_blob = f"thumbnails/{detection['date']}/{detection['id']}.jpg"
        detection['thumbnail_sas_url'] = generate_sas_url(thumbnail_blob)
        
        print(f"✓ Generated SAS URLs for {detection['id']}")
    
    # Save updated file with proper escaping
    output_file = 'public_website/api/detections_sas.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSAS URLs generated and saved to: {output_file}")
    print("\nExample SAS URL (valid for 30 days):")
    print(data['detections'][0]['video_sas_url'][:150] + '...')
    
    # Also generate SAS URLs for web files
    print("\nGenerating SAS URLs for web interface files...")
    web_files = {
        'index.html': generate_sas_url('index.html', days_valid=1),
        'css/style.css': generate_sas_url('css/style.css', days_valid=1),
        'js/app.js': generate_sas_url('js/app.js', days_valid=1),
        'api/detections.json': generate_sas_url('api/detections.json', days_valid=1)
    }
    
    print("\nWeb interface accessible at:")
    print(web_files['index.html'])
    
    return output_file

if __name__ == '__main__':
    update_detections_with_sas()