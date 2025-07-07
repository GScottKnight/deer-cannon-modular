#!/usr/bin/env python3
"""
Check CORS configuration on Azure Storage.
"""

import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure configuration
AZURE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

def check_cors():
    """Check current CORS configuration."""
    print("Connecting to Azure Storage...")
    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    
    try:
        props = blob_service.get_service_properties()
        print("\nCurrent CORS rules:")
        
        if props.get('cors') and len(props['cors']) > 0:
            for i, rule in enumerate(props['cors']):
                print(f"\nRule {i + 1}:")
                print(f"  Allowed origins: {rule.allowed_origins}")
                print(f"  Allowed methods: {rule.allowed_methods}")
                print(f"  Allowed headers: {rule.allowed_headers}")
                print(f"  Exposed headers: {rule.exposed_headers}")
                print(f"  Max age: {rule.max_age_in_seconds}")
        else:
            print("No CORS rules configured")
            
    except Exception as e:
        print(f"Error checking CORS: {e}")

if __name__ == '__main__':
    check_cors()