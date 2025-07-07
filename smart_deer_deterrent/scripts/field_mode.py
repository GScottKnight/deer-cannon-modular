#!/usr/bin/env python3
"""
Field Deployment Mode for Smart Deer Deterrent System

This script configures the system for field deployment with:
- Laptop screen management (dim/off)
- WiFi hotspot setup
- Headless operation
- Remote phone access
"""

import os
import sys
import time
import subprocess
import signal
import socket
import qrcode
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FieldDeployment:
    def __init__(self):
        self.hotspot_name = "DeerDeterrent"
        self.hotspot_password = "deerhunter2024"
        self.server_ip = "10.0.0.1"
        self.server_port = 5001
        self.flask_process = None
        self.original_brightness = None
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
    def cleanup(self, signum=None, frame=None):
        """Cleanup on exit."""
        print("\n🛑 Shutting down field mode...")
        
        # Stop Flask if running
        if self.flask_process:
            self.flask_process.terminate()
            
        # Restore screen brightness
        if self.original_brightness is not None:
            self.set_screen_brightness(self.original_brightness)
            
        # Stop hotspot
        self.stop_hotspot()
        
        print("✅ Cleanup complete")
        sys.exit(0)
        
    def get_screen_brightness(self):
        """Get current screen brightness (macOS)."""
        try:
            result = subprocess.run(
                ['brightness', '-l'], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                # Parse brightness value
                for line in result.stdout.split('\n'):
                    if 'display 0:' in line:
                        brightness = float(line.split('brightness ')[-1])
                        return brightness
        except:
            pass
        return None
        
    def set_screen_brightness(self, level):
        """Set screen brightness (0.0 to 1.0)."""
        try:
            subprocess.run(['brightness', str(level)], check=True)
            return True
        except:
            print("⚠️  Could not control screen brightness")
            print("   Install brightness: brew install brightness")
            return False
            
    def turn_screen_off(self):
        """Turn screen off or to minimum brightness."""
        self.original_brightness = self.get_screen_brightness()
        
        # Try to turn screen completely off
        try:
            # macOS specific - put display to sleep
            subprocess.run(['pmset', 'displaysleepnow'], check=True)
            print("✅ Screen turned off")
            return True
        except:
            # Fallback: set brightness to minimum
            if self.set_screen_brightness(0.0):
                print("✅ Screen brightness set to minimum")
                return True
                
        return False
        
    def prevent_sleep(self):
        """Prevent system from sleeping."""
        try:
            # macOS caffeinate command
            subprocess.Popen(['caffeinate', '-disu'])
            print("✅ Sleep prevention enabled")
            return True
        except:
            print("⚠️  Could not prevent sleep")
            return False
            
    def create_hotspot(self):
        """Create WiFi hotspot (requires admin privileges)."""
        print(f"\n📡 Creating WiFi Hotspot: {self.hotspot_name}")
        
        # Note: macOS doesn't have easy command-line hotspot creation
        # This is a placeholder - actual implementation would need:
        # 1. Internet Sharing via System Preferences
        # 2. Or use third-party tools
        
        print("⚠️  Automatic hotspot creation not available on macOS")
        print("\n📱 Manual Setup Required:")
        print("1. Open System Preferences > Sharing")
        print("2. Click 'Internet Sharing'")
        print("3. Share connection from: Wi-Fi")
        print("4. To computers using: Wi-Fi")
        print("5. Wi-Fi Options:")
        print(f"   - Network Name: {self.hotspot_name}")
        print(f"   - Password: {self.hotspot_password}")
        print("6. Check 'Internet Sharing' to start")
        
        return False
        
    def stop_hotspot(self):
        """Stop WiFi hotspot."""
        # Placeholder for macOS
        pass
        
    def get_local_ip(self):
        """Get local IP address."""
        try:
            # Create a dummy socket to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
            
    def generate_qr_code(self, url):
        """Generate QR code for easy phone access."""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        # Print QR code to terminal
        qr.print_ascii(invert=True)
        
        # Save QR code image
        img = qr.make_image(fill_color="black", back_color="white")
        img.save("field_mode_qr.png")
        print(f"\n💾 QR code saved to: field_mode_qr.png")
        
    def start_flask_app(self):
        """Start Flask app in background."""
        print("\n🚀 Starting Flask app...")
        
        # Path to main.py
        app_path = Path(__file__).parent.parent / "app" / "main.py"
        
        # Start Flask process
        self.flask_process = subprocess.Popen(
            [sys.executable, str(app_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for Flask to start
        time.sleep(5)
        
        # Check if running
        if self.flask_process.poll() is None:
            print("✅ Flask app started successfully")
            return True
        else:
            print("❌ Flask app failed to start")
            return False
            
    def display_connection_info(self):
        """Display connection information."""
        ip = self.get_local_ip()
        url = f"http://{ip}:{self.server_port}/mobile"
        
        print("\n" + "="*60)
        print("📱 MOBILE ACCESS INFORMATION")
        print("="*60)
        print(f"\n🌐 URL: {url}")
        print(f"\n📶 WiFi Network: {self.hotspot_name}")
        print(f"🔑 Password: {self.hotspot_password}")
        print("\n📷 Scan QR Code with your phone:")
        
        self.generate_qr_code(url)
        
        print("\n" + "="*60)
        
    def run(self, screen_delay=30, skip_hotspot=False, keep_screen_on=False):
        """Run field deployment mode."""
        print("\n🦌 SMART DEER DETERRENT - FIELD MODE")
        print("="*60)
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Prevent sleep
        self.prevent_sleep()
        
        # Create hotspot (unless skipped)
        if not skip_hotspot:
            self.create_hotspot()
            
        # Start Flask app
        if not self.start_flask_app():
            return
            
        # Display connection info
        self.display_connection_info()
        
        if not keep_screen_on:
            # Countdown before turning screen off
            print(f"\n⏱️  Screen will turn off in {screen_delay} seconds...")
            print("   Press Ctrl+C to cancel")
            
            try:
                for i in range(screen_delay, 0, -1):
                    print(f"   {i}...", end='\r')
                    time.sleep(1)
                    
                # Turn screen off
                print("\n🖥️  Turning screen off...")
                self.turn_screen_off()
                
            except KeyboardInterrupt:
                print("\n⏸️  Screen shutdown cancelled")
                
        # Keep running
        print("\n✅ Field mode active!")
        print("   Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
                # Could add system monitoring here
        except KeyboardInterrupt:
            pass
            
        self.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Field deployment mode for deer deterrent')
    parser.add_argument(
        '--screen-delay',
        type=int,
        default=30,
        help='Seconds before turning screen off (default: 30)'
    )
    parser.add_argument(
        '--skip-hotspot',
        action='store_true',
        help='Skip WiFi hotspot setup'
    )
    parser.add_argument(
        '--keep-screen-on',
        action='store_true',
        help='Keep screen on (for testing)'
    )
    
    args = parser.parse_args()
    
    # Check if running as admin (for some features)
    if os.geteuid() != 0 and not args.skip_hotspot:
        print("⚠️  Some features require administrator privileges")
        print("   Run with: sudo python field_mode.py")
        print("   Or use --skip-hotspot to skip WiFi setup")
    
    # Create and run field deployment
    field = FieldDeployment()
    field.run(
        screen_delay=args.screen_delay,
        skip_hotspot=args.skip_hotspot,
        keep_screen_on=args.keep_screen_on
    )


if __name__ == "__main__":
    main()