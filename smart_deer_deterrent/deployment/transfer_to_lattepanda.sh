#!/bin/bash
# Transfer script to copy Smart Deer Deterrent to LattePanda
# Run this from your Mac

# Configuration - UPDATE THESE
LATTEPANDA_USER="username"  # Change to your LattePanda username
LATTEPANDA_IP="192.168.1.XXX"  # Change to your LattePanda IP
SOURCE_DIR="/Users/gsknight/Documents/Deer_Cannon_Modular/smart_deer_deterrent"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Smart Deer Deterrent - Transfer to LattePanda${NC}"
echo "=============================================="

# Check if variables are updated
if [ "$LATTEPANDA_USER" == "username" ] || [ "$LATTEPANDA_IP" == "192.168.1.XXX" ]; then
    echo -e "${RED}ERROR: Please update LATTEPANDA_USER and LATTEPANDA_IP in this script${NC}"
    exit 1
fi

# Test SSH connection
echo -e "${YELLOW}Testing SSH connection to $LATTEPANDA_USER@$LATTEPANDA_IP...${NC}"
if ! ssh -o ConnectTimeout=5 "$LATTEPANDA_USER@$LATTEPANDA_IP" exit 2>/dev/null; then
    echo -e "${RED}ERROR: Cannot connect to LattePanda. Check IP and ensure SSH is enabled.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ SSH connection successful${NC}"

# Create deployment package
echo -e "${YELLOW}Creating deployment package...${NC}"
cd "$(dirname "$SOURCE_DIR")"
tar -czf deer_deterrent_deploy.tar.gz \
    --exclude='*/venv' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='*/uploads/*' \
    --exclude='*/detections/2*' \
    --exclude='*/logs/*.log' \
    --exclude='.git' \
    "$(basename "$SOURCE_DIR")/"

echo -e "${GREEN}✓ Package created${NC}"

# Transfer files
echo -e "${YELLOW}Transferring files to LattePanda...${NC}"
scp deer_deterrent_deploy.tar.gz "$LATTEPANDA_USER@$LATTEPANDA_IP:~/"

# Transfer deployment scripts separately for easy access
scp "$SOURCE_DIR/deployment/deploy_to_lattepanda.sh" "$LATTEPANDA_USER@$LATTEPANDA_IP:~/"
scp "$SOURCE_DIR/deployment/QUICK_START.md" "$LATTEPANDA_USER@$LATTEPANDA_IP:~/"

# Transfer model files
echo -e "${YELLOW}Transferring model files...${NC}"
scp "$SOURCE_DIR/app/models/"*.pt "$LATTEPANDA_USER@$LATTEPANDA_IP:~/"

# Clean up local package
rm deer_deterrent_deploy.tar.gz

echo -e "${GREEN}✓ Transfer complete!${NC}"
echo ""
echo "Next steps on LattePanda:"
echo "1. SSH into LattePanda: ssh $LATTEPANDA_USER@$LATTEPANDA_IP"
echo "2. Extract files: tar -xzf deer_deterrent_deploy.tar.gz"
echo "3. Run deployment: chmod +x deploy_to_lattepanda.sh && ./deploy_to_lattepanda.sh"
echo "4. Follow the QUICK_START.md guide"
echo ""
echo -e "${YELLOW}Tip: The QUICK_START.md file has been copied to your home directory on LattePanda${NC}"