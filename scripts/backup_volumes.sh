#!/bin/bash

# Tự động phát hiện project name từ docker-compose
get_project_name() {
    # Phương pháp 1: Từ docker-compose config
    PROJECT_FROM_CONFIG=$(docker-compose config --services 2>/dev/null | head -1)
    if [ ! -z "$PROJECT_FROM_CONFIG" ]; then
        # Lấy tên project từ container name pattern
        RUNNING_CONTAINER=$(docker-compose ps -q 2>/dev/null | head -1)
        if [ ! -z "$RUNNING_CONTAINER" ]; then
            CONTAINER_NAME=$(docker inspect --format='{{.Name}}' $RUNNING_CONTAINER 2>/dev/null | sed 's/\///')
            PROJECT_NAME=$(echo $CONTAINER_NAME | cut -d'_' -f1)
        fi
    fi
    
    # Phương pháp 2: Từ tên thư mục hiện tại (fallback)
    if [ -z "$PROJECT_NAME" ]; then
        PROJECT_NAME=$(basename $(pwd) | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
    fi
    
    # Phương pháp 3: Kiểm tra volumes thực tế có trong Docker
    if [ ! -z "$PROJECT_NAME" ]; then
        if docker volume ls | grep -q "${PROJECT_NAME}_app_data"; then
            echo $PROJECT_NAME
            return 0
        fi
    fi
    
    # Phương pháp 4: Tìm volumes có pattern _app_data
    DETECTED_VOLUME=$(docker volume ls --format "{{.Name}}" | grep "_app_data$" | head -1)
    if [ ! -z "$DETECTED_VOLUME" ]; then
        echo $(echo $DETECTED_VOLUME | sed 's/_app_data$//')
        return 0
    fi
    
    # Fallback cuối cùng
    echo "unknown_project"
}

# Phát hiện project name
PROJECT_NAME=$(get_project_name)
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}📦 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Tạo thư mục backup
mkdir -p $BACKUP_DIR

echo -e "${BLUE}🗄️  Docker Volume Backup Tool${NC}"
echo "================================================"
echo -e "${BLUE}🔍 Detected project name:${NC} $PROJECT_NAME"

# Kiểm tra volumes có tồn tại không
print_step "Checking existing volumes..."

DATA_VOLUME="${PROJECT_NAME}_app_data"
MODELS_VOLUME="${PROJECT_NAME}_app_models"

echo "  🔍 Looking for volume: $DATA_VOLUME"
if docker volume inspect $DATA_VOLUME >/dev/null 2>&1; then
    print_success "Found data volume: $DATA_VOLUME"
else
    print_warning "Volume $DATA_VOLUME not found"
    DATA_VOLUME=""
fi

echo "  🔍 Looking for volume: $MODELS_VOLUME"
if docker volume inspect $MODELS_VOLUME >/dev/null 2>&1; then
    print_success "Found models volume: $MODELS_VOLUME"
else
    print_warning "Volume $MODELS_VOLUME not found"
    MODELS_VOLUME=""
fi

# Hiển thị tất cả volumes để debug
echo ""
print_step "All available volumes:"
docker volume ls --format "table {{.Name}}\t{{.Driver}}" | grep -E "(app_data|app_models|NAME)"

if [ -z "$DATA_VOLUME" ] && [ -z "$MODELS_VOLUME" ]; then
    print_error "No target volumes found!"
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting:${NC}"
    echo "  1. Check if containers are created: docker-compose ps"
    echo "  2. Check actual volume names: docker volume ls"
    echo "  3. Try running docker-compose up first to create volumes"
    exit 1
fi

# ========================================
# BACKUP DATA VOLUME
# ========================================
if [ ! -z "$DATA_VOLUME" ]; then
    print_step "Backing up data volume: $DATA_VOLUME"
    
    # Kiểm tra volume có data không
    VOLUME_SIZE=$(docker run --rm -v $DATA_VOLUME:/data alpine du -sh /data 2>/dev/null | cut -f1)
    
    if [ ! -z "$VOLUME_SIZE" ] && [ "$VOLUME_SIZE" != "0" ]; then
        echo "  📊 Volume size: $VOLUME_SIZE"
        
        docker run --rm \
          -v $DATA_VOLUME:/data \
          -v $(pwd)/$BACKUP_DIR:/backup \
          alpine \
          sh -c "cd /data && if [ \"\$(ls -A .)\" ]; then tar czf /backup/app_data_${TIMESTAMP}.tar.gz . 2>/dev/null; else echo 'Empty directory'; fi"
        
        if [ -f "$BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz" ]; then
            BACKUP_SIZE=$(ls -lh $BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz 2>/dev/null | awk '{print $5}')
            print_success "Data volume backed up successfully ($BACKUP_SIZE)"
        else
            print_warning "Data volume appears to be empty"
            touch $BACKUP_DIR/app_data_${TIMESTAMP}_empty.marker
            print_success "Empty data volume marked"
        fi
    else
        print_warning "Data volume appears to be empty"
        touch $BACKUP_DIR/app_data_${TIMESTAMP}_empty.marker
        print_success "Empty data volume marked"
    fi
fi

# ========================================
# BACKUP MODELS VOLUME
# ========================================
if [ ! -z "$MODELS_VOLUME" ]; then
    print_step "Backing up models volume: $MODELS_VOLUME"
    
    # Kiểm tra volume có data không
    VOLUME_SIZE=$(docker run --rm -v $MODELS_VOLUME:/models alpine du -sh /models 2>/dev/null | cut -f1)
    
    if [ ! -z "$VOLUME_SIZE" ] && [ "$VOLUME_SIZE" != "0" ]; then
        echo "  📊 Volume size: $VOLUME_SIZE"
        
        docker run --rm \
          -v $MODELS_VOLUME:/models \
          -v $(pwd)/$BACKUP_DIR:/backup \
          alpine \
          sh -c "cd /models && if [ \"\$(ls -A .)\" ]; then tar czf /backup/app_models_${TIMESTAMP}.tar.gz . 2>/dev/null; else echo 'Empty directory'; fi"
        
        if [ -f "$BACKUP_DIR/app_models_${TIMESTAMP}.tar.gz" ]; then
            BACKUP_SIZE=$(ls -lh $BACKUP_DIR/app_models_${TIMESTAMP}.tar.gz 2>/dev/null | awk '{print $5}')
            print_success "Models volume backed up successfully ($BACKUP_SIZE)"
        else
            print_warning "Models volume appears to be empty"
            touch $BACKUP_DIR/app_models_${TIMESTAMP}_empty.marker
            print_success "Empty models volume marked"
        fi
    else
        print_warning "Models volume appears to be empty"
        touch $BACKUP_DIR/app_models_${TIMESTAMP}_empty.marker
        print_success "Empty models volume marked"
    fi
fi

# ========================================
# SUMMARY & CLEANUP
# ========================================
echo ""
print_step "Generating backup summary..."

# Tạo file summary
SUMMARY_FILE="$BACKUP_DIR/backup_summary_${TIMESTAMP}.txt"
echo "Docker Volume Backup Summary" > $SUMMARY_FILE
echo "============================" >> $SUMMARY_FILE
echo "Timestamp: $TIMESTAMP" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "Project: $PROJECT_NAME" >> $SUMMARY_FILE
echo "Data Volume: $DATA_VOLUME" >> $SUMMARY_FILE
echo "Models Volume: $MODELS_VOLUME" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

if [ -f "$BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz" ]; then
    DATA_SIZE=$(ls -lh $BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz | awk '{print $5}')
    echo "✅ Data backup: app_data_${TIMESTAMP}.tar.gz ($DATA_SIZE)" >> $SUMMARY_FILE
elif [ -f "$BACKUP_DIR/app_data_${TIMESTAMP}_empty.marker" ]; then
    echo "⚠️  Data backup: Empty volume (marked)" >> $SUMMARY_FILE
fi

if [ -f "$BACKUP_DIR/app_models_${TIMESTAMP}.tar.gz" ]; then
    MODELS_SIZE=$(ls -lh $BACKUP_DIR/app_models_${TIMESTAMP}.tar.gz | awk '{print $5}')
    echo "✅ Models backup: app_models_${TIMESTAMP}.tar.gz ($MODELS_SIZE)" >> $SUMMARY_FILE
elif [ -f "$BACKUP_DIR/app_models_${TIMESTAMP}_empty.marker" ]; then
    echo "⚠️  Models backup: Empty volume (marked)" >> $SUMMARY_FILE
fi

print_success "Backup summary saved: $SUMMARY_FILE"

# Cleanup old backups (giữ lại 5 bản gần nhất)
print_step "Cleaning up old backups (keeping latest 5)..."
ls -t $BACKUP_DIR/app_data_*.tar.gz 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
ls -t $BACKUP_DIR/app_data_*_empty.marker 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
ls -t $BACKUP_DIR/app_models_*.tar.gz 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
ls -t $BACKUP_DIR/app_models_*_empty.marker 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
ls -t $BACKUP_DIR/backup_summary_*.txt 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null

print_success "Old backups cleaned up"

# ========================================
# FINAL SUMMARY
# ========================================
echo ""
echo -e "${GREEN}🎉 Backup completed successfully!${NC}"
echo "================================================"
echo -e "${BLUE}🏷️  Project:${NC} $PROJECT_NAME"
echo -e "${BLUE}📁 Backup location:${NC} $BACKUP_DIR"
echo -e "${BLUE}🕐 Timestamp:${NC} $TIMESTAMP"
echo ""
echo -e "${BLUE}📋 Files created:${NC}"

if [ -f "$BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz" ]; then
    SIZE=$(ls -lh $BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz | awk '{print $5}')
    echo "  📦 app_data_${TIMESTAMP}.tar.gz ($SIZE)"
fi

if [ -f "$BACKUP_DIR/app_models_${TIMESTAMP}.tar.gz" ]; then
    SIZE=$(ls -lh $BACKUP_DIR/app_models_${TIMESTAMP}.tar.gz | awk '{print $5}')
    echo "  🧠 app_models_${TIMESTAMP}.tar.gz ($SIZE)"
fi

echo "  📄 backup_summary_${TIMESTAMP}.txt"

echo ""
echo -e "${BLUE}💡 Usage tips:${NC}"
echo "  • List backups: ls -la $BACKUP_DIR/"
echo "  • Restore: ./scripts/restore_volumes.sh $TIMESTAMP"
echo "  • Check summary: cat $BACKUP_DIR/backup_summary_${TIMESTAMP}.txt"
echo "  • Detected volumes: $DATA_VOLUME, $MODELS_VOLUME"