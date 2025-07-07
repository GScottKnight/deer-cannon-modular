# Making Your Detection Videos Publicly Accessible

## Option 1: Enable CORS on Azure (Recommended)

1. Go to Azure Portal: https://portal.azure.com
2. Navigate to your storage account: `deerdetections`
3. In the left menu, find "Resource sharing (CORS)"
4. Add a CORS rule:
   - Allowed origins: `*` (or specific domains)
   - Allowed methods: `GET, HEAD, OPTIONS`
   - Allowed headers: `*`
   - Exposed headers: `*`
   - Max age: `3600`

5. The web interface will be accessible at:
   https://deerdetections.blob.core.windows.net/deer-detections/index.html
   
   But first, you need to upload the web files:

## Upload Web Interface to Azure

Run this command to upload the web interface:

```bash
python upload_website_to_azure.py
```

## Option 2: Deploy to GitHub Pages (Free)

1. Create a new GitHub repository
2. Push the contents of `public_website/` to the repo
3. Enable GitHub Pages in repository settings
4. Access at: https://[your-username].github.io/[repo-name]

## Option 3: Deploy to Netlify (Free)

1. Drag and drop the `public_website` folder to netlify.com
2. Get instant URL like: https://amazing-deer-detector.netlify.app

## Current Direct Access

For now, you can view individual videos at these URLs:
- https://deerdetections.blob.core.windows.net/deer-detections/videos/2025-06-30/detection_20250630_180205.mp4
- https://deerdetections.blob.core.windows.net/deer-detections/videos/2025-06-30/detection_20250630_161951.mp4
- https://deerdetections.blob.core.windows.net/deer-detections/videos/2025-06-30/detection_20250630_111736.mp4
- https://deerdetections.blob.core.windows.net/deer-detections/videos/2025-06-30/detection_20250630_111508.mp4