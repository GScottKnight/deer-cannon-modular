#!/bin/bash
# Deployment script for Smart Deer Deterrent on LattePanda
# This script sets up the system on a fresh Ubuntu installation

set -e  # Exit on error

echo "Smart Deer Deterrent - LattePanda Deployment Script"
echo "=================================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root. Please run as a regular user with sudo privileges."
   exit 1
fi

# Update system
echo "1. Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required system packages
echo "2. Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-opencv \
    git \
    nginx \
    ffmpeg \
    v4l-utils \
    htop \
    iotop

# Create user for the service
echo "3. Creating service user..."
if ! id "deer-detector" &>/dev/null; then
    sudo useradd -r -s /bin/bash -m -d /home/deer-detector deer-detector
    sudo usermod -a -G video deer-detector  # Add to video group for camera access
fi

# Setup application directory
echo "4. Setting up application..."
APP_DIR="/home/deer-detector/smart_deer_deterrent"

# Option 1: Git repository (uncomment and update if using Git)
# REPO_URL="https://github.com/yourusername/smart_deer_deterrent.git"
# if [ -d "$APP_DIR" ]; then
#     echo "Repository exists, pulling latest changes..."
#     cd "$APP_DIR"
#     sudo -u deer-detector git pull
# else
#     echo "Cloning repository..."
#     sudo -u deer-detector git clone "$REPO_URL" "$APP_DIR"
# fi

# Option 2: Manual copy (if not using Git)
if [ ! -d "$APP_DIR" ]; then
    echo "Creating application directory..."
    sudo mkdir -p "$APP_DIR"
    sudo chown -R deer-detector:deer-detector "$APP_DIR"
    echo "Please copy the smart_deer_deterrent folder to $APP_DIR"
    echo "You can use: rsync -avz /path/to/smart_deer_deterrent/ $APP_DIR/"
fi

# Create Python virtual environment
echo "5. Setting up Python environment..."
cd "$APP_DIR"
sudo -u deer-detector python3 -m venv venv

# Install Python dependencies
echo "6. Installing Python dependencies..."
sudo -u deer-detector venv/bin/pip install --upgrade pip
sudo -u deer-detector venv/bin/pip install -r requirements.txt

# Create necessary directories
echo "7. Creating directories..."
sudo -u deer-detector mkdir -p app/logs
sudo -u deer-detector mkdir -p app/uploads
sudo -u deer-detector mkdir -p app/static
sudo -u deer-detector mkdir -p app/detections

# Create log directory for systemd
sudo mkdir -p /var/log/deer-detector
sudo chown deer-detector:deer-detector /var/log/deer-detector

# Install systemd service
echo "8. Installing systemd service..."
sudo cp deployment/deer-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable deer-detector.service

# Configure nginx (optional - for reverse proxy)
echo "9. Configuring nginx..."
sudo cp deployment/nginx.conf /etc/nginx/sites-available/deer-detector
sudo ln -sf /etc/nginx/sites-available/deer-detector /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

# Set up firewall
echo "10. Configuring firewall..."
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 5001/tcp  # Flask app (if not using nginx)
sudo ufw --force enable

# Create startup script
echo "11. Creating helper scripts..."
cat > /home/deer-detector/start_deer_detector.sh << 'EOF'
#!/bin/bash
sudo systemctl start deer-detector
echo "Smart Deer Deterrent started. Check status with: sudo systemctl status deer-detector"
EOF

cat > /home/deer-detector/stop_deer_detector.sh << 'EOF'
#!/bin/bash
sudo systemctl stop deer-detector
echo "Smart Deer Deterrent stopped."
EOF

cat > /home/deer-detector/view_logs.sh << 'EOF'
#!/bin/bash
echo "=== Recent Application Logs ==="
tail -n 50 /home/deer-detector/smart_deer_deterrent/app/logs/app.log
echo -e "\n=== Recent Detection Logs ==="
tail -n 20 /home/deer-detector/smart_deer_deterrent/app/logs/detections.log
echo -e "\n=== Service Logs ==="
sudo journalctl -u deer-detector -n 50
EOF

chmod +x /home/deer-detector/*.sh
sudo chown deer-detector:deer-detector /home/deer-detector/*.sh

# Performance optimizations for LattePanda
echo "12. Applying performance optimizations..."

# Increase camera buffer size
echo "options uvcvideo nodrop=1 timeout=6000" | sudo tee /etc/modprobe.d/uvcvideo.conf

# Set CPU governor to performance (optional)
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Configure system for headless operation
echo "13. Configuring for headless operation..."
sudo systemctl set-default multi-user.target

echo ""
echo "=================================================="
echo "Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Update the repository URL in this script"
echo "2. Copy your model files (best.pt, yolov8n.pt) to app/models/"
echo "3. Start the service: sudo systemctl start deer-detector"
echo "4. Check status: sudo systemctl status deer-detector"
echo "5. View logs: /home/deer-detector/view_logs.sh"
echo "6. Access web interface: http://$(hostname -I | awk '{print $1}'):5001"
echo ""
echo "For remote access via SSH:"
echo "ssh $(whoami)@$(hostname -I | awk '{print $1}')"
echo "=================================================="