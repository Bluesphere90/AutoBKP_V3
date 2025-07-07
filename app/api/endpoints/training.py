# app/api/endpoints/training.py

import logging
import shutil
import time
from pathlib import Path
from typing import Optional # Đảm bảo đã import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Path as FastApiPath, Query

# --- Project Imports ---
from app.api import schemas
from app.ml.models import train_client_models
from app.ml.utils import get_client_data_path, get_client_models_path
import app.core.config as config

import json
import os

# --- Setup Logger and Router ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# --- Helper Function ---
def _save_uploaded_file(upload_file: UploadFile, destination: Path):
    """Lưu file được upload vào đường dẫn đích."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        logger.info(f"File '{upload_file.filename}' đã được lưu vào '{destination}'")
    finally:
        upload_file.file.close()

# --- Background Task Runner (Sửa thứ tự tham số) ---
def _run_training_task(client_id: str, initial_model_type_str: Optional[str] = None): # client_id (không default) lên trước
    """Hàm thực thi việc huấn luyện (để chạy nền)."""
    action = "Huấn luyện ban đầu" if initial_model_type_str else "Huấn luyện tăng cường"
    log_message = f"[Background Task] Bắt đầu {action} cho client: {client_id}"
    if initial_model_type_str:
        log_message += f" với model '{initial_model_type_str}'"
    logger.info(log_message)
    try:
        # Gọi train_client_models với đúng tên tham số
        success = train_client_models(
            client_id=client_id,
            initial_model_type_str=initial_model_type_str
        )
        if success:
            logger.info(f"[Background Task] {action} thành công cho client: {client_id}")
        else:
            logger.error(f"[Background Task] {action} thất bại cho client: {client_id}.")
    except Exception as e:
        logger.error(f"[Background Task] Lỗi nghiêm trọng trong quá trình {action} client {client_id}: {e}", exc_info=True)

# --- API Endpoints ---

@router.post(
    "/training/{client_id}",
    response_model=schemas.MessageResponse,
    status_code=202,
    summary="Huấn luyện mô hình lần đầu cho client (chọn model)",
    description="Upload file CSV huấn luyện ban đầu, **chỉ định loại mô hình**, và kích hoạt huấn luyện nền.",
    responses={
        202: {"description": "Yêu cầu huấn luyện đã được chấp nhận và đang chạy nền."},
        400: {"model": schemas.ErrorResponse,
              "description": "Định dạng file không hợp lệ, loại model không hợp lệ, hoặc lỗi upload."},
        422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input (ví dụ: thiếu file)."},
        # FastAPI tự xử lý
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server khi lưu file hoặc bắt đầu huấn luyện."}
    }
)
async def train_initial_model(
    background_tasks: BackgroundTasks,
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
    model_type: config.SupportedModels = Query(
        default=config.DEFAULT_MODEL_TYPE,
        description=f"Loại mô hình muốn huấn luyện. Lựa chọn: {', '.join([e.value for e in config.SupportedModels])}"
    ),
    # 4. Tham số File (CÓ default)
    file: UploadFile = File(..., description="File CSV chứa dữ liệu huấn luyện ban đầu...")
    # ---------------------------------
):
    """Endpoint huấn luyện ban đầu, người dùng chỉ định model."""
    if not file.filename or not file.filename.endswith(".csv"):
        logger.error(f"Client {client_id}: Tên file không hợp lệ hoặc không phải .csv ('{file.filename}').")
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ hoặc định dạng file không phải .csv")

    model_type_str = model_type.value
    logger.info(f"Client {client_id}: Yêu cầu huấn luyện ban đầu với model_type='{model_type_str}'")

    client_data_path = get_client_data_path(client_id)
    client_model_path = get_client_models_path(client_id)

    initial_data_file = client_data_path / config.TRAINING_DATA_FILENAME
    metadata_files = list(client_model_path.glob(f"{config.METADATA_FILENAME_PREFIX}*.json"))

    if initial_data_file.exists() or metadata_files:
        logger.warning(f"Client {client_id}: Model already exists. Rejecting initial training request.")
        raise HTTPException(
            status_code=409,  # 409 Conflict
            detail=f"Model for client '{client_id}' already exists. Use incremental training or choose a different client_id."
        )

    destination = client_data_path / config.TRAINING_DATA_FILENAME

    try:
        _save_uploaded_file(file, destination)
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi khi lưu file upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu file upload: {e}")

    # Gọi add_task với đúng thứ tự của _run_training_task
    logger.info(f"Client {client_id}: Thêm task huấn luyện ban đầu (model: {model_type_str}) vào background.")
    background_tasks.add_task(
        _run_training_task,
        client_id=client_id,                    # client_id trước
        initial_model_type_str=model_type_str   # initial_model_type_str sau
    )

    return schemas.MessageResponse(
        message=f"Đã nhận file và bắt đầu huấn luyện mô hình '{model_type_str}' cho client '{client_id}'. Quá trình chạy nền.",
        client_id=client_id,
        status_code=202
    )


@router.post(
    "/training/incremental/{client_id}",
    response_model=schemas.MessageResponse,
    status_code=202, # Accepted
    summary="Huấn luyện tăng cường mô hình cho client",
    description="Upload file CSV dữ liệu mới và kích hoạt huấn luyện tăng cường (sử dụng model đã chọn trước đó).",
    tags=["Training"], # Thêm tag
     # --- Phần responses đầy đủ ---
     responses={
        202: {"description": "Yêu cầu huấn luyện tăng cường đã được chấp nhận và đang chạy nền."},
        400: {"model": schemas.ErrorResponse, "description": "Định dạng file không hợp lệ hoặc lỗi upload."},
        404: {"model": schemas.ErrorResponse, "description": "Chưa có dữ liệu/metadata huấn luyện ban đầu cho client."},
        422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input (ví dụ: thiếu file)."},
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server khi lưu file hoặc bắt đầu huấn luyện."}
    }
    # -----------------------------
)
async def train_incremental_model(
    # Tham số hàm (đã sửa thứ tự)
    background_tasks: BackgroundTasks,
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
    file: UploadFile = File(..., description="File CSV chứa dữ liệu huấn luyện mới (có thể có các cột khác)")
):
    """
    Endpoint để upload dữ liệu mới và bắt đầu huấn luyện tăng cường.
    Sẽ tự động sử dụng loại model đã được lưu trong metadata.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        logger.error(f"Client {client_id}: Tên file không hợp lệ hoặc không phải .csv ('{file.filename}').")
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ hoặc định dạng file không phải .csv")

    client_data_path = get_client_data_path(client_id)
    client_models_path = get_client_models_path(client_id)
    initial_data_file = client_data_path / config.TRAINING_DATA_FILENAME
    metadata_files = sorted(client_models_path.glob(f"{config.METADATA_FILENAME_PREFIX}*.json"), reverse=True)

    if not metadata_files and not initial_data_file.exists():
            logger.error(f"Client {client_id}: Không tìm thấy dữ liệu huấn luyện ban đầu hoặc metadata. Không thể huấn luyện tăng cường.")
            raise HTTPException(
                status_code=404,
                detail=f"Chưa có dữ liệu/metadata huấn luyện ban đầu cho client '{client_id}'. Vui lòng huấn luyện lần đầu trước."
            )

    timestamp = int(time.time())
    incremental_filename = f"{config.INCREMENTAL_DATA_PREFIX}{timestamp}.csv"
    destination = client_data_path / incremental_filename

    try:
        _save_uploaded_file(file, destination)
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi khi lưu file incremental upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu file incremental upload: {e}")

    # Gọi add_task với đúng thứ tự của _run_training_task
    logger.info(f"Client {client_id}: Thêm task huấn luyện tăng cường vào background.")
    background_tasks.add_task(
        _run_training_task,
        client_id=client_id,
        initial_model_type_str=None # Luôn là None cho incremental
    )

    return schemas.MessageResponse(
        message=f"Đã nhận file và bắt đầu huấn luyện tăng cường cho client '{client_id}'. Quá trình chạy nền.",
        client_id=client_id,
        status_code=202
    )

@router.get(
    "/status/{client_id}",
    response_model=schemas.TrainingStatusResponse,
    summary="Kiểm tra tình trạng training của client",
    description="Đọc file metadata mới nhất để xác định tình trạng training hiện tại.",
    tags=["Training"],
    responses={
        200: {"description": "Thông tin tình trạng training"},
        404: {"model": schemas.ErrorResponse, "description": "Không tìm thấy thông tin training cho client"}
    }
)
async def get_training_status(
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc")
) -> schemas.TrainingStatusResponse:
    """Endpoint kiểm tra tình trạng training đơn giản."""
    
    client_models_path = get_client_models_path(client_id)
    
    # Tìm file metadata mới nhất
    metadata_files = sorted(
        client_models_path.glob(f"{config.METADATA_FILENAME_PREFIX}*.json"),
        key=os.path.getmtime,
        reverse=True
    )
    
    if not metadata_files:
        raise HTTPException(
            status_code=404, 
            detail=f"Không tìm thấy thông tin training cho client '{client_id}'"
        )
    
    latest_metadata_file = metadata_files[0]
    
    try:
        with open(latest_metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return schemas.TrainingStatusResponse(
            client_id=client_id,
            status=metadata.get("status", "UNKNOWN"),
            training_type=metadata.get("training_type"),
            selected_model_type=metadata.get("selected_model_type"),
            training_timestamp=metadata.get("training_timestamp_utc"),
            duration_seconds=metadata.get("training_duration_seconds"),
            error_message=metadata.get("error_message")
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi đọc metadata file {latest_metadata_file}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi đọc thông tin training: {e}"
        )