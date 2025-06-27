#!/bin/bash

# Tự động phát hiện project name (giống function trong backup script)
get_project_name() {
    # Phương pháp 1: Từ docker-compose config
    PROJECT_FROM_CONFIG=$(docker-compose config --services 2>/dev/null | head -1)
    if [ ! -z "$PROJECT_FROM_CONFIG" ]; then
        RUNNING_CONTAINER=$(docker-compose ps -q 2>/dev/null | head -1)
        if [ ! -z "$RUNNING_CONTAINER" ]; then
            CONTAINER_NAME=$(docker inspect --format='{{.Name}}' $RUNNING_CONTAINER 2>/dev/null | sed 's/\///')
            PROJECT_NAME=$(echo $CONTAINER_NAME | cut -d'_' -f1)
        fi
    fi
    
    # Phương pháp 2: Từ tên thư mục hiện tại
    if [ -z "$PROJECT_NAME" ]; then
        PROJECT_NAME=$(basename $(pwd) | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
    fi
    
    # Phương pháp 3: Kiểm tra volumes thực tế
    if [ ! -z "$PROJECT_NAME" ]; then
        if docker volume ls | grep -q "${PROJECT_NAME}_app_data"; then
            echo $PROJECT_NAME
            return 0
        fi
    fi
    
    # Phương pháp 4: Tìm volumes có pattern
    DETECTED_VOLUME=$(docker volume ls --format "{{.Name}}" | grep "_app_data$" | head -1)
    if [ ! -z "$DETECTED_VOLUME" ]; then
        echo $(echo $DETECTED_VOLUME | sed 's/_app_data$//')
        return 0
    fi
    
    echo "unknown_project"
}

PROJECT_NAME=$(get_project_name)
BACKUP_DIR="./backups"

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

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

echo -e "${BLUE}🔄 Docker Volume Restore Tool${NC}"
echo "================================================"
echo -e "${BLUE}🔍 Detected project name:${NC} $PROJECT_NAME"

# Kiểm tra tham số
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}📋 Available backups:${NC}"
    echo ""
    if [ -d "$BACKUP_DIR" ]; then
        ls -la $BACKUP_DIR/ | grep -E "(app_data_|app_models_|backup_summary_)" | sort -r
    else
        echo "No backup directory found at: $BACKUP_DIR"
    fi
    echo ""
    echo "Usage: $0 <timestamp>"
    echo "Example: $0 20250115_143022"
    echo ""
    echo -e "${BLUE}🔍 Current volumes:${NC}"
    docker volume ls --format "table {{.Name}}\t{{.Driver}}" | grep -E "(${PROJECT_NAME}_|NAME)"
    exit 1
fi

TIMESTAMP=$1

print_step "Restoring volumes from backup: $TIMESTAMP"
echo -e "${BLUE}🎯 Target volumes:${NC} ${PROJECT_NAME}_app_data, ${PROJECT_NAME}_app_models"

# Kiểm tra files backup có tồn tại không
DATA_BACKUP="$BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz"
MODELS_BACKUP="$BACKUP_DIR/app_models_${TIMESTAMP}.tar.gz"
DATA_EMPTY_MARKER="$BACKUP_DIR/app_data_${TIMESTAMP}_empty.marker"
MODELS_EMPTY_MARKER="$BACKUP_DIR/app_models_${TIMESTAMP}_empty.marker"

if [ ! -f "$DATA_BACKUP" ] && [ ! -f "$DATA_EMPTY_MARKER" ] && [ ! -f "$MODELS_BACKUP" ] && [ ! -f "$MODELS_EMPTY_MARKER" ]; then
    print_error "No backup files found for timestamp: $TIMESTAMP"
    echo ""
    echo "Available backups:"
    ls -la $BACKUP_DIR/ | grep -E "${TIMESTAMP}"
    exit 1
fi

# Hiển thị thông tin backup sẽ restore
echo ""
echo -e "${BLUE}📋 Backup files found:${NC}"
[ -f "$DATA_BACKUP" ] && echo "  📦 $(ls -lh $DATA_BACKUP | awk '{print $9 " (" $5 ")"}')"
[ -f "$MODELS_BACKUP" ] && echo "  🧠 $(ls -lh $MODELS_BACKUP | awk '{print $9 " (" $5 ")"}')"
[ -f "$DATA_EMPTY_MARKER" ] && echo "  📦 Data volume was empty"
[ -f "$MODELS_EMPTY_MARKER" ] && echo "  🧠 Models volume was empty"

# Cảnh báo
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will overwrite current volume data!${NC}"
echo -e "${YELLOW}📂 Target volumes:${NC}"
echo "   • ${PROJECT_NAME}_app_data"
echo "   • ${PROJECT_NAME}_app_models"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled."
    exit 1
fi

# Restore data volume
if [ -f "$DATA_BACKUP" ]; then
    print_step "Restoring data volume: ${PROJECT_NAME}_app_data"
    docker run --rm \
      -v ${PROJECT_NAME}_app_data:/data \
      -v $(pwd)/$BACKUP_DIR:/backup \
      alpine \
      sh -c "cd /data && rm -rf * .* 2>/dev/null || true; tar xzf /backup/app_data_${TIMESTAMP}.tar.gz -C /data 2>/dev/null"
    
    if [ $? -eq 0 ]; then
        print_success "Data volume restored successfully"
    else
        print_error "Data volume restore failed"
    fi
elif [ -f "$DATA_EMPTY_MARKER" ]; then
    print_step "Clearing data volume (was empty in backup)..."
    docker run --rm \
      -v ${PROJECT_NAME}_app_data:/data \
      alpine \
      sh -c "cd /data && rm -rf * .* 2>/dev/null || true"
    print_success "Data volume cleared"
fi

# Restore models volume
if [ -f "$MODELS_BACKUP" ]; then
    print_step "Restoring models volume: ${PROJECT_NAME}_app_models"
    docker run --rm \
      -v ${PROJECT_NAME}_app_models:/models \
      -v $(pwd)/$BACKUP_DIR:/backup \
      alpine \
      sh -c "cd /models && rm -rf * .* 2>/dev/null || true; tar xzf /backup/app_models_${TIMESTAMP}.tar.gz -C /models 2>/dev/null"
    
    if [ $? -eq 0 ]; then
        print_success "Models volume restored successfully"
    else
        print_error "Models volume restore failed"
    fi
elif [ -f "$MODELS_EMPTY_MARKER" ]; then
    print_step "Clearing models volume (was empty in backup)..."
    docker run --rm \
      -v ${PROJECT_NAME}_app_models:/models \
      alpine \
      sh -c "cd /models && rm -rf * .* 2>/dev/null || true"
    print_success "Models volume cleared"
fi

echo ""
print_success "Volume restore completed!"
echo ""
echo -e "${BLUE}💡 Next steps:${NC}"
echo "  • Restart containers: docker-compose restart"
echo "  • Check data: docker-compose exec api_server ls -la /app/data"
echo "  • Check models: docker-compose exec api_server ls -la /app/models"
echo "  • Verify API: curl http://localhost:8000/docs"