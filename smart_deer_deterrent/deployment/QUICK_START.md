# Quick Start Guide - LattePanda Deployment

## From Your Mac (Current Machine)

### 1. Stop Current Flask App
```bash
# Find and kill the current process
lsof -i :5001
kill -9 <PID>
```

### 2. Package Files for Transfer
```bash
cd /Users/gsknight/Documents/Deer_Cannon_Modular

# Create deployment package (excluding large/unnecessary files)
tar -czf deer_deterrent_deploy.tar.gz \
  --exclude='*/venv' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*/uploads/*' \
  --exclude='*/detections/*' \
  smart_deer_deterrent/
```

### 3. Transfer to LattePanda
```bash
# Via SSH (replace with your LattePanda details)
scp deer_deterrent_deploy.tar.gz username@lattepanda_ip:~/

# Also copy model files separately if large
scp smart_deer_deterrent/app/models/*.pt username@lattepanda_ip:~/
```

## On LattePanda

### 1. Initial Setup (First Time Only)
```bash
# Extract files
tar -xzf deer_deterrent_deploy.tar.gz

# Run deployment script
cd smart_deer_deterrent/deployment
chmod +x deploy_to_lattepanda.sh
./deploy_to_lattepanda.sh

# Copy to service directory
sudo cp -r ~/smart_deer_deterrent /home/deer-detector/
sudo chown -R deer-detector:deer-detector /home/deer-detector/smart_deer_deterrent

# Copy model files
sudo cp ~/*.pt /home/deer-detector/smart_deer_deterrent/app/models/
```

### 2. Quick Test Before Service
```bash
# Test camera detection
v4l2-ctl --list-devices

# Test Flask app directly
cd /home/deer-detector/smart_deer_deterrent
sudo -u deer-detector venv/bin/python app/main.py
# Press Ctrl+C to stop
```

### 3. Start Service
```bash
# Start the service
sudo systemctl start deer-detector

# Enable auto-start on boot
sudo systemctl enable deer-detector

# Check status
sudo systemctl status deer-detector

# View live logs
sudo journalctl -u deer-detector -f
```

### 4. Access Web Interface
- From LattePanda: http://localhost:5001
- From your Mac: http://lattepanda_ip:5001
- Status Dashboard: http://lattepanda_ip:5001/status
- Logs: http://lattepanda_ip:5001/logs

## Quick Commands Reference

```bash
# Service control
sudo systemctl start deer-detector
sudo systemctl stop deer-detector
sudo systemctl restart deer-detector
sudo systemctl status deer-detector

# View logs
sudo journalctl -u deer-detector -n 100
tail -f /home/deer-detector/smart_deer_deterrent/app/logs/app.log

# Check camera
v4l2-ctl -d /dev/video0 --all

# Monitor resources
htop
df -h
```

## If Things Go Wrong

1. **Camera not found**: Check `ls -la /dev/video*` and update camera index
2. **Port already in use**: `sudo lsof -i :5001` and kill the process
3. **Permission denied**: Check file ownership with `ls -la`
4. **Service won't start**: Run manually to see errors: `sudo -u deer-detector venv/bin/python app/main.py`

## Remote Monitoring

Once running, you can monitor from your Mac:
- SSH: `ssh username@lattepanda_ip`
- Web: `http://lattepanda_ip:5001`
- Copy detection videos: `scp username@lattepanda_ip:/home/deer-detector/smart_deer_deterrent/app/detections/* ./`