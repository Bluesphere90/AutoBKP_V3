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
from app.ml.database import (
    create_training_table,
    insert_training_records,
    load_all_training_data,
    get_training_data_count,
    database_exists
)
import app.core.config as config

from typing import List, Dict, Any

import json
import os

import pandas as pd
from app.ml.database import (
    create_training_table,
    insert_training_records,
    get_training_data_count
)



# --- Setup Logger and Router ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# --- Helper Function ---
def _parse_csv_to_records(upload_file: UploadFile) -> List[Dict[str, Any]]:
    """Parse CSV file và convert sang list of dicts để insert vào database."""
    try:
        # Read CSV content
        content = upload_file.file.read()
        upload_file.file.seek(0)  # Reset file pointer

        # Parse with pandas
        df = pd.read_csv(
            pd.io.common.StringIO(content.decode('utf-8-sig')),
            sep=None,
            engine='python',
            skipinitialspace=True
        )

        if df.empty:
            logger.warning(f"CSV file '{upload_file.filename}' is empty")
            return []

        # Convert to records (list of dicts)
        records = df.to_dict('records')

        # Clean records - remove NaN values
        cleaned_records = []
        for record in records:
            cleaned_record = {}
            for key, value in record.items():
                # Convert NaN to empty string
                if pd.isna(value):
                    cleaned_record[key] = ""
                else:
                    cleaned_record[key] = str(value).strip()
            cleaned_records.append(cleaned_record)

        logger.info(f"Parsed {len(cleaned_records)} records from CSV '{upload_file.filename}'")
        return cleaned_records

    except Exception as e:
        logger.error(f"Failed to parse CSV file '{upload_file.filename}': {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {e}"
        )


def _validate_model_name(model_name: str):
    """Validate model_name parameter."""
    if model_name not in config.VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name '{model_name}'. Must be one of: {config.VALID_MODEL_NAMES}"
        )


def _find_latest_metadata_file(models_path: Path) -> Optional[Path]:
    """Tìm file metadata mới nhất trong thư mục models."""
    metadata_files = sorted(
        models_path.glob(f"{config.METADATA_FILENAME_PREFIX}*.json"),
        key=os.path.getmtime,
        reverse=True
    )
    return metadata_files[0] if metadata_files else None

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
def _run_training_task(client_id: str, model_name: str, initial_model_type_str: Optional[str] = None):
    """Background task để chạy training."""
    action = "New training" if initial_model_type_str else "Retrain"
    logger.info(f"[Background Task] Starting {action} for client: {client_id}, model: {model_name}")

    try:
        # Call updated train_client_models function
        success = train_client_models(
            client_id=client_id,
            model_name=model_name,
            initial_model_type_str=initial_model_type_str
        )

        if success:
            logger.info(f"[Background Task] {action} successful for client: {client_id}, model: {model_name}")
        else:
            logger.error(f"[Background Task] {action} failed for client: {client_id}, model: {model_name}")

    except Exception as e:
        logger.error(f"[Background Task] Error during {action} for client {client_id}, model {model_name}: {e}",
                     exc_info=True)

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


def _parse_csv_to_records(upload_file: UploadFile) -> List[Dict[str, Any]]:
    """Parse CSV file và convert sang list of dicts."""
    try:
        # Read CSV content
        content = upload_file.file.read()
        upload_file.file.seek(0)  # Reset file pointer

        # Parse with pandas
        df = pd.read_csv(
            pd.io.common.StringIO(content.decode('utf-8-sig')),
            sep=None,
            engine='python',
            skipinitialspace=True
        )

        if df.empty:
            logger.warning(f"CSV file '{upload_file.filename}' is empty")
            return []

        # Convert to records và clean NaN values
        records = []
        for _, row in df.iterrows():
            record = {}
            for col, value in row.items():
                if pd.isna(value):
                    record[col] = ""
                else:
                    record[col] = str(value).strip()
            records.append(record)

        logger.info(f"Parsed {len(records)} records from CSV '{upload_file.filename}'")
        return records

    except Exception as e:
        logger.error(f"Failed to parse CSV file '{upload_file.filename}': {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")


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


@router.post(
    "/test-database/{client_id}",
    response_model=schemas.MessageResponse,
    summary="[TEST] Upload CSV và insert vào database",
    description="Endpoint test để thử nghiệm database integration"
)
async def test_database_upload(
        client_id: str = FastApiPath(..., description="Client ID", example="client_abc"),
        file: UploadFile = File(..., description="CSV file")
):
    """
    Endpoint đơn giản để test database integration.
    Chỉ upload CSV → parse → insert vào database.
    """

    # Validate file
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    logger.info(f"Testing database upload for client: {client_id}")

    try:
        # Parse CSV to records
        records = _parse_csv_to_records(file)

        if not records:
            raise HTTPException(status_code=400, detail="CSV file is empty or invalid")

        # Create database table if not exists
        table_created = create_training_table(client_id)
        if not table_created:
            raise HTTPException(status_code=500, detail="Failed to create database table")

        # Insert records
        inserted_count = insert_training_records(client_id, records)
        total_count = get_training_data_count(client_id)

        logger.info(f"Client {client_id}: Inserted {inserted_count}/{len(records)} records. Total: {total_count}")

        return schemas.MessageResponse(
            message=f"Successfully processed CSV for client '{client_id}'. Inserted {inserted_count} new records. Total records: {total_count}",
            client_id=client_id,
            status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database upload failed for client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database operation failed: {e}")


# Endpoint chính cho training với database
@router.post(
    "/train/{client_id}",
    response_model=schemas.MessageResponse,
    status_code=202,
    summary="Flexible Training API",
    description="Train model với hoặc không có CSV file. Nếu có CSV thì import vào DB trước khi train."
)
async def flexible_train(
        background_tasks: BackgroundTasks,
        client_id: str = FastApiPath(..., description="Client ID", example="client_abc"),
        model_name: str = Query(..., description="Model name: IN or OUT", example="IN"),
        model_type: Optional[config.SupportedModels] = Query(
            None,
            description=f"Model type (required for new training): {', '.join([e.value for e in config.SupportedModels])}"
        ),
        file: Optional[UploadFile] = File(None,
                                          description="Optional CSV file. If not provided, train with existing database.")
):
    """
    Flexible training workflow:

    Case 1: Có CSV file
    - Parse CSV và insert vào database
    - Train với toàn bộ data trong database

    Case 2: Không có CSV file
    - Kiểm tra database có data không
    - Train với data hiện tại trong database

    Case 3: Không có CSV + database rỗng/không tồn tại
    - Trả về lỗi
    """

    # Validate model name
    _validate_model_name(model_name)

    logger.info(f"Flexible training request for client: {client_id}, model: {model_name}")
    logger.info(f"CSV file provided: {'Yes' if file else 'No'}")

    # Check current database status
    from app.ml.database import database_exists, get_training_data_count
    db_exists = database_exists(client_id)
    current_record_count = get_training_data_count(client_id) if db_exists else 0

    logger.info(f"Current database status - Exists: {db_exists}, Records: {current_record_count}")

    # Case 1: CSV file provided - import to database first
    if file is not None:
        logger.info("Case 1: CSV file provided, importing to database")

        # Validate file
        if not file.filename or not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")

        try:
            # Parse CSV to records
            records = _parse_csv_to_records(file)

            if not records:
                raise HTTPException(status_code=400, detail="CSV file is empty or invalid")

            # Ensure database table exists
            table_created = create_training_table(client_id)
            if not table_created:
                raise HTTPException(status_code=500, detail="Failed to create database table")

            # Insert records into database
            inserted_count = insert_training_records(client_id, records)
            new_total_records = get_training_data_count(client_id)

            logger.info(f"CSV import: {inserted_count}/{len(records)} new records inserted. Total: {new_total_records}")
            csv_info = f"Imported {inserted_count} new records from CSV. "

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to process CSV file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {e}")

    # Case 2: No CSV file - check if database has data
    else:
        logger.info("Case 2: No CSV file, checking existing database")
        csv_info = ""

        if not db_exists or current_record_count == 0:
            # Case 3: No CSV + no database data = error
            logger.error(f"No training data available for client {client_id}")
            raise HTTPException(
                status_code=400,
                detail=f"No training data found for client '{client_id}'. Please provide a CSV file or ensure database contains training data."
            )

        logger.info(f"Using existing database with {current_record_count} records")
        new_total_records = current_record_count

    # At this point, we know database has data - proceed with training

    # Check if this is new training or retrain
    models_path = get_client_models_path(client_id)
    existing_metadata_files = list(models_path.glob(f"{config.METADATA_FILENAME_PREFIX}{model_name}_*.json"))
    is_new_training = len(existing_metadata_files) == 0

    # Determine model_type to use
    model_type_str = None

    if is_new_training:
        # New training - require model_type
        if model_type is None:
            raise HTTPException(
                status_code=400,
                detail=f"model_type is required for new training of model '{model_name}'"
            )
        model_type_str = model_type.value
        logger.info(f"New training for model '{model_name}' with type: {model_type_str}")

    else:
        # Retrain - use existing model_type
        if model_type is not None:
            logger.info(f"Ignoring model_type parameter for retrain of existing model '{model_name}'")

        # Load existing model_type from latest metadata
        latest_metadata = sorted(existing_metadata_files, key=os.path.getmtime, reverse=True)[0]
        try:
            with open(latest_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            model_type_str = metadata.get("selected_model_type") or config.DEFAULT_MODEL_TYPE.value
        except Exception as e:
            logger.warning(f"Failed to read existing metadata: {e}, using default")
            model_type_str = config.DEFAULT_MODEL_TYPE.value

        logger.info(f"Retrain for model '{model_name}' using existing type: {model_type_str}")

    # Start training in background
    logger.info(f"Starting background training for client: {client_id}, model: {model_name}")
    background_tasks.add_task(
        _run_training_task,
        client_id=client_id,
        model_name=model_name,
        initial_model_type_str=model_type_str if is_new_training else None
    )

    action = "New training" if is_new_training else "Retrain"

    return schemas.MessageResponse(
        message=f"{action} started for client '{client_id}', model '{model_name}'. {csv_info}Total records: {new_total_records}. Training running in background.",
        client_id=client_id,
        status_code=202
    )

@router.post(
    "/test-database-with-model/{client_id}",
    response_model=schemas.MessageResponse,
    summary="[TEST] Upload CSV với model name support",
    description="Test database + model name integration"
)
async def test_database_with_model(
        client_id: str = FastApiPath(..., description="Client ID", example="client_abc"),
        model_name: str = Query(..., description="Model name: IN or OUT", example="IN"),
        file: UploadFile = File(..., description="CSV file")
):
    """
    Test endpoint với model_name support.
    Database vẫn per-client, nhưng có thể train nhiều models cho 1 client.
    """

    # Validate inputs
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    _validate_model_name(model_name)

    logger.info(f"Testing database upload for client: {client_id}, model: {model_name}")

    try:
        # Parse CSV to records
        records = _parse_csv_to_records(file)

        if not records:
            raise HTTPException(status_code=400, detail="CSV file is empty or invalid")

        # Create database table if not exists (still per-client)
        table_created = create_training_table(client_id)
        if not table_created:
            raise HTTPException(status_code=500, detail="Failed to create database table")

        # Insert records (same database, different models will use same data)
        inserted_count = insert_training_records(client_id, records)
        total_count = get_training_data_count(client_id)

        logger.info(
            f"Client {client_id}, Model {model_name}: Inserted {inserted_count}/{len(records)} records. Total: {total_count}")

        return schemas.MessageResponse(
            message=f"Successfully processed CSV for client '{client_id}', model '{model_name}'. Inserted {inserted_count} new records. Total records: {total_count}",
            client_id=client_id,
            status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database upload failed for client {client_id}, model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Database operation failed: {e}")