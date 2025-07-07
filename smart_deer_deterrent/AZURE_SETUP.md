# Azure Setup Guide for Deer Detection System

This guide will help you set up Azure Blob Storage to automatically upload and share your deer detection videos.

## Prerequisites

- Microsoft Azure account
- Azure CLI (optional but recommended)

## Step 1: Create Azure Storage Account

### Using Azure Portal:

1. Go to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource" → "Storage" → "Storage account"
3. Configure:
   - **Resource group**: Create new or use existing
   - **Storage account name**: Choose unique name (e.g., `deerdetections`)
   - **Region**: Choose closest to you
   - **Performance**: Standard
   - **Redundancy**: LRS (Locally-redundant storage) for cost savings
4. Review and create

### Using Azure CLI:

```bash
# Create resource group
az group create --name deer-detection-rg --location eastus

# Create storage account
az storage account create \
  --name deerdetections \
  --resource-group deer-detection-rg \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2
```

## Step 2: Get Storage Credentials

### Using Azure Portal:

1. Go to your storage account
2. Under "Security + networking" → "Access keys"
3. Copy:
   - Storage account name
   - Key1 (or Key2)
   - Connection string

### Using Azure CLI:

```bash
# Get storage account key
az storage account keys list \
  --account-name deerdetections \
  --resource-group deer-detection-rg \
  --query "[0].value" -o tsv
```

## Step 3: Create Container

### Using Azure Portal:

1. In your storage account, go to "Data storage" → "Containers"
2. Click "+ Container"
3. Name: `deer-detections`
4. Public access level: "Blob" (allows public read of videos)

### Using Azure CLI:

```bash
# Set environment variables
export AZURE_STORAGE_ACCOUNT=deerdetections
export AZURE_STORAGE_KEY=your_key_here

# Create container
az storage container create \
  --name deer-detections \
  --public-access blob
```

## Step 4: Configure CORS (for web access)

### Using Azure Portal:

1. In storage account, go to "Resource sharing (CORS)"
2. Under "Blob service", add:
   - Allowed origins: `*`
   - Allowed methods: ``
   - Allowed headers: `*`
   - Exposed headers: `*`
   - Max age: `86400`

### Using Azure CLI:

```bash
az storage cors add \
  --services b \
  --methods GET HEAD OPTIONS \
  --origins '*' \
  --allowed-headers '*' \
  --exposed-headers '*' \
  --max-age 86400
```

## Step 5: Configure Local Environment

1. Copy `.env.template` to `.env`:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` with your credentials:
   ```
   AZURE_STORAGE_ACCOUNT_NAME=deerdetections
   AZURE_STORAGE_ACCOUNT_KEY=your_key_here
   AZURE_CONTAINER_NAME=deer-detections
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Step 6: Test the Uploader

1. Run a test upload:
   ```bash
   python azure_uploader.py
   ```

2. Check Azure Portal to see if videos appear in your container

## Step 7: Deploy Web Interface

### Option A: Azure Static Web Apps (Recommended)

1. Push `public_website/` to a GitHub repository
2. In Azure Portal:
   - Create new "Static Web App"
   - Link to your GitHub repo
   - Set app location: `/public_website`
   - Azure will auto-deploy on commits

### Option B: GitHub Pages (Free)

1. Push to GitHub
2. Enable GitHub Pages in repo settings
3. Update `app.js` with your storage account URL

### Option C: Netlify (Free)

1. Drag `public_website/` folder to Netlify
2. Update `app.js` with your storage account URL

## Step 8: Run Upload Service

### Manual:
```bash
./scripts/start_uploader.sh
```

### As a service (Linux):
```bash
# Copy and edit the service file
sudo cp scripts/deer-uploader.service /etc/systemd/system/
sudo nano /etc/systemd/system/deer-uploader.service  # Update paths
sudo systemctl enable deer-uploader
sudo systemctl start deer-uploader
```

### As a LaunchAgent (macOS):
```bash
# Copy to LaunchAgents
cp scripts/com.deerdeterrent.uploader.plist ~/Library/LaunchAgents/
# Edit the file to update paths
launchctl load ~/Library/LaunchAgents/com.deerdeterrent.uploader.plist
```

## Cost Estimation

- **Storage**: ~$0.02 per GB/month
- **Bandwidth**: ~$0.09 per GB (outbound)
- **Transactions**: ~$0.005 per 10,000 operations

For typical usage (100 videos/day, 10MB each):
- Monthly storage: ~30GB = $0.60
- Monthly bandwidth: ~5GB = $0.45
- **Total: ~$1-2/month**

## Troubleshooting

1. **Permission Denied**: Check storage account key
2. **Container not found**: Ensure container name matches .env
3. **CORS errors**: Configure CORS settings in Azure
4. **Videos not playing**: Check blob public access level

## Security Notes

- Never commit `.env` file to git
- Consider using SAS tokens for more restricted access
- Rotate storage keys periodically
- Monitor usage to avoid unexpected costs