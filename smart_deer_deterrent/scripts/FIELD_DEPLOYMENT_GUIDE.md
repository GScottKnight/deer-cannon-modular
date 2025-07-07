# Field Deployment Guide

This guide helps you deploy the Smart Deer Deterrent System in the field with your laptop.

## Quick Start

```bash
# Basic field mode (screen turns off after 30 seconds)
python scripts/field_mode.py --skip-hotspot

# Keep screen on for testing
python scripts/field_mode.py --skip-hotspot --keep-screen-on

# Custom screen delay
python scripts/field_mode.py --skip-hotspot --screen-delay 60
```

## Step-by-Step Setup

### 1. Prepare Your Laptop

**Battery Settings:**
- Go to System Preferences > Battery
- Set "Turn display off after" to Never (while on battery)
- Uncheck "Put hard disks to sleep when possible"
- Uncheck "Enable Power Nap"

**Network Setup:**
- Turn off WiFi sleep: `sudo pmset -b tcpkeepalive 1`

### 2. Create WiFi Hotspot (Manual)

Since macOS doesn't allow command-line hotspot creation, set it up manually:

1. **Open System Preferences > Sharing**
2. Click **"Internet Sharing"** (don't check it yet)
3. Configure:
   - Share your connection from: **Ethernet** (or your internet source)
   - To computers using: **Wi-Fi**
4. Click **"Wi-Fi Options..."**
   - Network Name: `DeerDeterrent`
   - Channel: Leave default
   - Security: WPA2 Personal
   - Password: `deerhunter2024`
5. Check **"Internet Sharing"** to start

### 3. Run Field Mode

```bash
cd /Users/gsknight/Documents/Deer_Cannon_Modular/smart_deer_deterrent
python scripts/field_mode.py --skip-hotspot
```

The script will:
- Start the Flask app
- Show QR code for phone access
- Display connection URL
- Turn screen off after 30 seconds

### 4. Connect Your Phone

1. Connect to WiFi: `DeerDeterrent` 
2. Password: `deerhunter2024`
3. Open browser to: `http://10.0.0.1:5001/mobile`
   - Or scan the QR code displayed

## Field Deployment Checklist

### Before Leaving:
- [ ] Laptop fully charged
- [ ] Test camera with cover removed
- [ ] Test mobile interface connection
- [ ] Disable all notifications
- [ ] Close unnecessary apps
- [ ] Set volume to mute

### Equipment:
- [ ] Laptop with power adapter
- [ ] USB camera(s)
- [ ] External battery pack (optional)
- [ ] Weatherproof enclosure/cover
- [ ] IR illuminator (for night)
- [ ] Mounting hardware

### In the Field:
1. **Position Equipment:**
   - Mount camera with clear view
   - Keep laptop in weatherproof location
   - Ensure good ventilation for cooling

2. **Power Management:**
   - Use external battery if no AC power
   - Monitor battery level via mobile interface
   - System auto-shuts down at 10% battery

3. **Start System:**
   ```bash
   python scripts/field_mode.py --skip-hotspot
   ```

4. **Verify Operation:**
   - Connect phone to hotspot
   - Check live video feed
   - Verify detection is working
   - Test emergency stop button

## Troubleshooting

### Can't Connect to Mobile Interface
1. Check WiFi hotspot is active
2. Verify IP address: `ifconfig en0`
3. Try: `http://[laptop-ip]:5001/mobile`

### Screen Won't Turn Off
- Install brightness tool: `brew install brightness`
- Or manually: Cmd+Shift+Eject (older Macs)
- Or: close lid partially (if using external camera)

### System Sleeping Despite caffeinate
- Check Energy Saver settings
- Run: `sudo pmset -g`
- Disable all sleep: `sudo pmset -a sleep 0`

### Camera Not Working
- Check USB connection
- Try different USB port
- Verify with: `ls /dev/video*`
- Test with Photo Booth app

## Security Notes

1. **Change default password** in `field_mode.py`
2. **Use WPA2** for hotspot security
3. **Limit hotspot range** if possible
4. **Monitor access** - check connected devices

## Advanced Options

### Run Without Display
```bash
# SSH into laptop from phone
ssh user@10.0.0.1

# Start in screen session
screen -S deer
python scripts/field_mode.py --skip-hotspot

# Detach: Ctrl+A, D
# Reattach: screen -r deer
```

### Automatic Startup
Create `/Users/[username]/Library/LaunchAgents/com.deer.deterrent.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.deer.deterrent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/path/to/field_mode.py</string>
        <string>--skip-hotspot</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load with: `launchctl load ~/Library/LaunchAgents/com.deer.deterrent.plist`

## Emergency Procedures

### Remote Shutdown
From phone browser:
1. Go to mobile interface
2. Press red STOP button
3. Confirm emergency stop

### Physical Access
If screen is off:
1. Press any key to wake screen
2. Enter password if locked
3. Ctrl+C in terminal to stop

### Hard Reset
- Hold power button for 10 seconds
- System will stop but won't save data