# LattePanda Deployment Checklist

## Pre-Deployment (On Development Machine)

### 1. Prepare Git Repository
- [ ] Commit all current changes
- [ ] Push to GitHub repository (you'll need to create one)
- [ ] Update `deploy_to_lattepanda.sh` with your GitHub URL (line 43)

### 2. Package Model Files
- [ ] Locate model files:
  - `/Users/gsknight/Documents/Deer_Cannon_Modular/smart_deer_deterrent/app/models/best.pt`
  - `/Users/gsknight/Documents/Deer_Cannon_Modular/smart_deer_deterrent/app/models/yolov8n.pt`
- [ ] Copy to USB drive or prepare for network transfer

### 3. Test Current System
- [ ] Verify camera 0 (ArduCam) is working
- [ ] Test detection is functioning properly
- [ ] Note current camera settings/index

## LattePanda Setup

### 1. Initial System Setup
- [ ] Install Ubuntu 22.04 LTS on LattePanda
- [ ] Connect to network (Ethernet recommended)
- [ ] Enable SSH: `sudo systemctl enable ssh`
- [ ] Note IP address: `hostname -I`

### 2. Transfer Files
```bash
# From your Mac, transfer deployment script
scp deployment/deploy_to_lattepanda.sh username@lattepanda_ip:~/

# Or if not using Git, transfer entire project
rsync -avz --exclude='venv' --exclude='*.pyc' --exclude='__pycache__' \
  /Users/gsknight/Documents/Deer_Cannon_Modular/smart_deer_deterrent/ \
  username@lattepanda_ip:~/smart_deer_deterrent/
```

### 3. Run Deployment Script
```bash
# SSH into LattePanda
ssh username@lattepanda_ip

# Make script executable
chmod +x deploy_to_lattepanda.sh

# Run deployment
./deploy_to_lattepanda.sh
```

### 4. Manual Steps After Script
- [ ] Copy model files to `/home/deer-detector/smart_deer_deterrent/app/models/`
- [ ] Verify camera detection: `v4l2-ctl --list-devices`
- [ ] Update camera index if needed in the config

### 5. Start and Test Service
```bash
# Start the service
sudo systemctl start deer-detector

# Check status
sudo systemctl status deer-detector

# View logs
sudo journalctl -u deer-detector -f

# Test web interface
# From another machine: http://lattepanda_ip:5001
```

## Post-Deployment Configuration

### 1. Camera Configuration
- [ ] Test ArduCam is detected as camera 0
- [ ] Verify auto-adjustment is working
- [ ] Check detection accuracy

### 2. Remote Access Setup
- [ ] Configure port forwarding on router for external access (optional)
- [ ] Set up dynamic DNS if needed (optional)
- [ ] Test monitoring dashboard: http://lattepanda_ip:5001/status

### 3. Performance Tuning
- [ ] Monitor CPU/Memory usage via dashboard
- [ ] Adjust `frame_skip` in MODEL_CONFIG if needed
- [ ] Check detection video saving is working

### 4. Security (Optional but Recommended)
- [ ] Change default passwords
- [ ] Configure firewall rules
- [ ] Set up nginx basic auth for web interface
- [ ] Use HTTPS with Let's Encrypt

## Troubleshooting Commands

```bash
# Check camera availability
ls -la /dev/video*
v4l2-ctl --list-devices

# Test camera directly
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 test.jpg

# Monitor system resources
htop
iotop

# Check service logs
sudo journalctl -u deer-detector --since "10 minutes ago"

# Restart service
sudo systemctl restart deer-detector

# Check Python errors
cd /home/deer-detector/smart_deer_deterrent
sudo -u deer-detector venv/bin/python app/main.py
```

## Monitoring from Development Machine

Once deployed, you can monitor the system remotely:

1. **Web Interface**: http://lattepanda_ip:5001
2. **Status Dashboard**: http://lattepanda_ip:5001/status
3. **Logs Viewer**: http://lattepanda_ip:5001/logs
4. **SSH Access**: `ssh username@lattepanda_ip`

## Backup Important Info

Record these for future reference:
- LattePanda IP: ________________
- SSH Username: ________________
- Camera Index: ________________
- Service Status Command: `sudo systemctl status deer-detector`
- Logs Location: `/home/deer-detector/smart_deer_deterrent/app/logs/`

## Notes
- The system will auto-start on boot once configured
- Detection videos are saved to `/home/deer-detector/smart_deer_deterrent/app/detections/`
- The deployment script creates helper scripts in `/home/deer-detector/`:
  - `start_deer_detector.sh`
  - `stop_deer_detector.sh`
  - `view_logs.sh`