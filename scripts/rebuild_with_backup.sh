#!/bin/bash

# Cấu hình

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
SERVICE_NAME="api_server"
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}🔄 $1${NC}"
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

print_step "Starting rebuild process with backup/restore..."

# ========================================
# BƯỚC 1: Dừng containers
# ========================================
print_step "Step 1: Stopping containers..."
docker-compose down
if [ $? -eq 0 ]; then
    print_success "Containers stopped successfully"
else
    print_error "Failed to stop containers"
    exit 1
fi

# ========================================
# BƯỚC 2: Backup current image
# ========================================
print_step "Step 2: Backing up current image..."

# Lấy tên image hiện tại
CURRENT_IMAGE=$(docker-compose config | grep "image:" | head -1 | awk '{print $2}' 2>/dev/null)

# Nếu không có image từ config, tạo tên image từ project
if [ -z "$CURRENT_IMAGE" ]; then
    CURRENT_IMAGE="${PROJECT_NAME}_${SERVICE_NAME}"
fi

# Kiểm tra xem image có tồn tại không
if docker image inspect $CURRENT_IMAGE >/dev/null 2>&1; then
    BACKUP_IMAGE_NAME="${CURRENT_IMAGE}_backup_${TIMESTAMP}"
    
    print_step "Tagging current image as backup: $BACKUP_IMAGE_NAME"
    docker tag $CURRENT_IMAGE $BACKUP_IMAGE_NAME
    
    if [ $? -eq 0 ]; then
        print_success "Image backed up as: $BACKUP_IMAGE_NAME"
        
        # Tùy chọn: Export image ra file
        print_step "Exporting image to file..."
        docker save $BACKUP_IMAGE_NAME | gzip > "$BACKUP_DIR/image_backup_${TIMESTAMP}.tar.gz"
        
        if [ $? -eq 0 ]; then
            print_success "Image exported to: $BACKUP_DIR/image_backup_${TIMESTAMP}.tar.gz"
        else
            print_warning "Failed to export image to file"
        fi
    else
        print_error "Failed to create backup tag"
        exit 1
    fi
else
    print_warning "No existing image found to backup. This might be the first build."
    BACKUP_IMAGE_NAME=""
fi

# ========================================
# BƯỚC 3: Backup Volumes (Data & Models)
# ========================================
print_step "Step 3: Backing up volumes..."

# Backup app_data volume
print_step "Backing up app_data volume..."
docker run --rm \
  -v ${PROJECT_NAME}_app_data:/data \
  -v $(pwd)/$BACKUP_DIR:/backup \
  alpine \
  tar czf /backup/app_data_${TIMESTAMP}.tar.gz -C /data . 2>/dev/null

if [ $? -eq 0 ]; then
    print_success "app_data volume backed up"
else
    print_warning "app_data volume backup failed (might be empty)"
fi

# Backup app_models volume
print_step "Backing up app_models volume..."
docker run --rm \
  -v ${PROJECT_NAME}_app_models:/models \
  -v $(pwd)/$BACKUP_DIR:/backup \
  alpine \
  tar czf /backup/app_models_${TIMESTAMP}.tar.gz -C /models . 2>/dev/null

if [ $? -eq 0 ]; then
    print_success "app_models volume backed up"
else
    print_warning "app_models volume backup failed (might be empty)"
fi

# ========================================
# BƯỚC 4: Build new image
# ========================================
print_step "Step 4: Building new image..."
docker-compose build --no-cache

if [ $? -eq 0 ]; then
    print_success "New image built successfully"
else
    print_error "Build failed! Attempting to restore..."
    
    # Restore previous image nếu có
    if [ ! -z "$BACKUP_IMAGE_NAME" ]; then
        print_step "Restoring previous image..."
        docker tag $BACKUP_IMAGE_NAME $CURRENT_IMAGE
        print_warning "Previous image restored. Please check your code changes."
    fi
    exit 1
fi

# ========================================
# BƯỚC 5: Start new containers
# ========================================
print_step "Step 5: Starting new containers..."
docker-compose up -d

if [ $? -eq 0 ]; then
    print_success "New containers started successfully"
else
    print_error "Failed to start new containers! Attempting to restore..."
    
    # Restore previous image nếu có
    if [ ! -z "$BACKUP_IMAGE_NAME" ]; then
        print_step "Restoring previous image..."
        docker tag $BACKUP_IMAGE_NAME $CURRENT_IMAGE
        docker-compose up -d
        print_warning "Previous image restored and started."
    fi
    exit 1
fi

# ========================================
# BƯỚC 6: Health check
# ========================================
print_step "Step 6: Performing health check..."
sleep 10

# Check if container is running
if docker-compose ps | grep -q "Up"; then
    print_success "Container is running"
    
    # Check API health
    if curl -f http://localhost:8000/docs >/dev/null 2>&1; then
        print_success "API is responding"
    else
        print_warning "API not responding yet, check logs: docker-compose logs"
    fi
else
    print_error "Container failed to start properly"
    docker-compose ps
    exit 1
fi

# ========================================
# BƯỚC 7: Verify volumes
# ========================================
print_step "Step 7: Verifying volumes are mounted..."
docker-compose exec -T api_server ls -la /app/data >/dev/null 2>&1 && print_success "Data volume mounted" || print_warning "Data volume issue"
docker-compose exec -T api_server ls -la /app/models >/dev/null 2>&1 && print_success "Models volume mounted" || print_warning "Models volume issue"

# ========================================
# BƯỚC 8: Cleanup old backup images (optional)
# ========================================
print_step "Step 8: Cleaning up old backup images..."

# Giữ lại 3 backup images gần nhất
OLD_BACKUPS=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep "${CURRENT_IMAGE}_backup_" | tail -n +4)

if [ ! -z "$OLD_BACKUPS" ]; then
    echo "$OLD_BACKUPS" | xargs docker rmi >/dev/null 2>&1
    print_success "Old backup images cleaned up"
fi

# ========================================
# SUMMARY
# ========================================
echo ""
print_success "🎉 Rebuild completed successfully!"
echo ""
echo -e "${BLUE}📊 Summary:${NC}"
echo "  • Backup image: $BACKUP_IMAGE_NAME"
echo "  • Data backup: $BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz"
echo "  • Models backup: $BACKUP_DIR/app_models_${TIMESTAMP}.tar.gz"
echo "  • Image file backup: $BACKUP_DIR/image_backup_${TIMESTAMP}.tar.gz"
echo ""
echo -e "${BLUE}🔗 Access points:${NC}"
echo "  • API Docs: http://localhost:8000/docs"
echo "  • Health check: curl http://localhost:8000/"
echo ""
echo -e "${BLUE}📋 Useful commands:${NC}"
echo "  • Check logs: docker-compose logs -f"
echo "  • Check status: docker-compose ps"
echo "  • Test API: curl http://localhost:8000/training/status/client_abc"