#!/bin/bash

# XTRADING Backup Script for External Drive
# This script creates a clean copy of your bot files to an external drive

echo "====================================="
echo "XTRADING Backup to External Drive"
echo "====================================="

# Check if external drive path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/external/drive"
    echo "Example: $0 /Volumes/MyExternalDrive/XTRADING_BACKUP"
    exit 1
fi

EXTERNAL_PATH="$1"
SOURCE_DIR="/Users/royaltyvixion/Documents/XTRADING"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${EXTERNAL_PATH}/XTRADING_${TIMESTAMP}"

echo ""
echo "Source: $SOURCE_DIR"
echo "Destination: $BACKUP_DIR"
echo ""

# Step 1: Kill any running tail processes that might be holding files
echo "Step 1: Stopping file monitoring processes..."
pkill -f "tail.*XTRADING" 2>/dev/null || true
sleep 1

# Step 2: Create backup directory
echo "Step 2: Creating backup directory..."
if ! mkdir -p "$BACKUP_DIR"; then
    echo "Error: Cannot create backup directory. Check if external drive is mounted and writable."
    exit 1
fi

# Step 3: List of important directories and files to copy
echo "Step 3: Copying essential bot files..."

# Create directory structure
mkdir -p "$BACKUP_DIR"/{es_bot,cl_bot,nq_bot,utils,configs,scripts,logs,web_platform,patterns,shared,trading_bot,orchestrator,tests}

# Copy Python bot files
echo "  - Copying bot implementations..."
cp -r "$SOURCE_DIR"/es_bot/* "$BACKUP_DIR"/es_bot/ 2>/dev/null || true
cp -r "$SOURCE_DIR"/cl_bot/* "$BACKUP_DIR"/cl_bot/ 2>/dev/null || true

# Copy main scripts
echo "  - Copying main scripts..."
cp "$SOURCE_DIR"/*.py "$BACKUP_DIR"/ 2>/dev/null || true
cp "$SOURCE_DIR"/*.sh "$BACKUP_DIR"/ 2>/dev/null || true

# Copy utilities
echo "  - Copying utilities..."
cp -r "$SOURCE_DIR"/utils/* "$BACKUP_DIR"/utils/ 2>/dev/null || true

# Copy configs
echo "  - Copying configurations..."
cp -r "$SOURCE_DIR"/configs/* "$BACKUP_DIR"/configs/ 2>/dev/null || true

# Copy scripts
echo "  - Copying scripts..."
cp -r "$SOURCE_DIR"/scripts/* "$BACKUP_DIR"/scripts/ 2>/dev/null || true

# Copy patterns
echo "  - Copying pattern files..."
cp -r "$SOURCE_DIR"/patterns/* "$BACKUP_DIR"/patterns/ 2>/dev/null || true

# Copy shared modules
echo "  - Copying shared modules..."
cp -r "$SOURCE_DIR"/shared/* "$BACKUP_DIR"/shared/ 2>/dev/null || true

# Copy trading bot
echo "  - Copying trading bot..."
cp -r "$SOURCE_DIR"/trading_bot/* "$BACKUP_DIR"/trading_bot/ 2>/dev/null || true

# Copy orchestrator
echo "  - Copying orchestrator..."
cp -r "$SOURCE_DIR"/orchestrator/* "$BACKUP_DIR"/orchestrator/ 2>/dev/null || true

# Copy web platform (excluding node_modules and large files)
echo "  - Copying web platform..."
rsync -av --exclude='node_modules' --exclude='__pycache__' --exclude='*.pyc' \
    "$SOURCE_DIR"/web_platform/ "$BACKUP_DIR"/web_platform/ 2>/dev/null || true

# Copy tests
echo "  - Copying tests..."
cp -r "$SOURCE_DIR"/tests/* "$BACKUP_DIR"/tests/ 2>/dev/null || true

# Copy environment files
echo "  - Copying environment files..."
cp "$SOURCE_DIR"/.env* "$BACKUP_DIR"/ 2>/dev/null || true

# Copy requirements
echo "  - Copying requirements..."
cp "$SOURCE_DIR"/requirements.txt "$BACKUP_DIR"/ 2>/dev/null || true
cp "$SOURCE_DIR"/package*.json "$BACKUP_DIR"/ 2>/dev/null || true

# Copy documentation
echo "  - Copying documentation..."
cp "$SOURCE_DIR"/*.md "$BACKUP_DIR"/ 2>/dev/null || true
cp "$SOURCE_DIR"/*.txt "$BACKUP_DIR"/ 2>/dev/null || true

# Copy Docker files if they exist
echo "  - Copying Docker files..."
cp "$SOURCE_DIR"/Dockerfile* "$BACKUP_DIR"/ 2>/dev/null || true
cp "$SOURCE_DIR"/docker-compose*.yml "$BACKUP_DIR"/ 2>/dev/null || true

# Create a minimal log backup (last 100 lines of each log)
echo "Step 4: Creating log snapshots..."
mkdir -p "$BACKUP_DIR"/logs/snapshots
for log in "$SOURCE_DIR"/logs/*.log; do
    if [ -f "$log" ]; then
        filename=$(basename "$log")
        tail -n 100 "$log" > "$BACKUP_DIR/logs/snapshots/${filename}.snapshot" 2>/dev/null || true
    fi
done

# Step 5: Create backup info file
echo "Step 5: Creating backup information file..."
cat > "$BACKUP_DIR/BACKUP_INFO.txt" << EOF
XTRADING Backup Information
===========================
Backup Date: $(date)
Source Directory: $SOURCE_DIR
Backup Directory: $BACKUP_DIR

Files Backed Up:
- All Python bot implementations (es_bot, cl_bot, nq_bot)
- All utility modules
- Configuration files
- Pattern discovery modules
- Web platform code (excluding node_modules)
- Scripts and orchestrator
- Environment files
- Requirements and dependencies
- Documentation files
- Log snapshots (last 100 lines)

To restore:
1. Copy this backup directory to your desired location
2. Install dependencies: pip install -r requirements.txt
3. Configure environment variables in .env files
4. Run verification: python verify_prerequisites.py

Note: Full log files were not copied to save space.
If you need complete logs, manually copy the logs/ directory.
EOF

# Step 6: Calculate backup size
echo ""
echo "Step 6: Calculating backup size..."
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "Backup size: $BACKUP_SIZE"

# Step 7: Verify backup
echo ""
echo "Step 7: Verifying backup..."
FILE_COUNT=$(find "$BACKUP_DIR" -type f | wc -l | tr -d ' ')
echo "Total files backed up: $FILE_COUNT"

echo ""
echo "====================================="
echo "✅ Backup completed successfully!"
echo "Location: $BACKUP_DIR"
echo "Size: $BACKUP_SIZE"
echo "Files: $FILE_COUNT"
echo "====================================="
echo ""
echo "You can now safely eject your external drive."
echo "To use this backup on another system, copy the entire"
echo "directory and follow the instructions in BACKUP_INFO.txt"