# app/api/endpoints/models.py

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Path as FastApiPath

# --- Project Imports ---
from app.api import schemas
from app.ml.utils import get_client_models_path
import app.core.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()


def _get_latest_metadata_file(client_models_path: Path) -> Optional[Path]:
    """Tìm file metadata mới nhất (copy từ training.py)."""
    metadata_files = sorted(
        client_models_path.glob(f"{config.METADATA_FILENAME_PREFIX}*.json"),
        key=os.path.getmtime,
        reverse=True
    )
    return metadata_files[0] if metadata_files else None


def _check_model_files_exist(client_models_path: Path) -> Dict[str, bool]:
    """Kiểm tra file models có tồn tại không."""
    return {
        "hachtoan_available": (client_models_path / config.HACHTOAN_MODEL_FILENAME).exists(),
        "mahanghoa_available": (client_models_path / config.MAHANGHOA_MODEL_FILENAME).exists(),
    }


@router.get(
    "/",
    response_model=schemas.ModelsListResponse,
    summary="Lấy danh sách tất cả models",
    description="Scan tất cả client folders và trả về thông tin models.",
    tags=["Models"]
)
async def list_all_models() -> schemas.ModelsListResponse:
    """Endpoint lấy danh sách tất cả models."""
    models_list = []

    if not config.BASE_MODELS_PATH.exists():
        return schemas.ModelsListResponse(models=[])

    # Scan tất cả client folders
    for client_dir in config.BASE_MODELS_PATH.iterdir():
        if not client_dir.is_dir():
            continue

        client_id = client_dir.name

        try:
            # Tìm metadata mới nhất
            latest_metadata_file = _get_latest_metadata_file(client_dir)
            if not latest_metadata_file:
                continue

            # Đọc metadata
            with open(latest_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Kiểm tra model files
            model_status = _check_model_files_exist(client_dir)

            # Tạo model info
            model_info = schemas.ModelInfo(
                client_id=client_id,
                model_type=metadata.get("selected_model_type", "Unknown"),
                training_type=metadata.get("training_type", "unknown"),
                status=metadata.get("status", "UNKNOWN"),
                created_at=metadata.get("training_timestamp_utc"),
                last_trained=metadata.get("training_timestamp_utc"),  # Tạm thời same as created
                hachtoan_available=model_status["hachtoan_available"],
                mahanghoa_available=model_status["mahanghoa_available"],
                sample_count=metadata.get("data_info", {}).get("total_samples_loaded", 0)
            )

            models_list.append(model_info)

        except Exception as e:
            logger.warning(f"Lỗi khi đọc metadata cho client {client_id}: {e}")
            continue

    return schemas.ModelsListResponse(models=models_list)


@router.get(
    "/{client_id}/status",
    response_model=schemas.TrainingStatusResponse,
    summary="Kiểm tra trạng thái model của client",
    description="Alias của /training/status/{client_id} - sử dụng cùng logic.",
    tags=["Models"]
)
async def get_model_status(
        client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc")
) -> schemas.TrainingStatusResponse:
    """Endpoint kiểm tra model status - tái sử dụng logic từ training endpoint."""
    # Import function từ training endpoint (tránh duplicate code)
    from app.api.endpoints.training import get_training_status
    return await get_training_status(client_id)


@router.get(
    "/{client_id}/metadata",
    response_model=schemas.ModelMetadataResponse,
    summary="Lấy chi tiết metadata của model",
    description="Trả về metadata đầy đủ (đã filter sensitive info).",
    tags=["Models"]
)
async def get_model_metadata(
        client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc")
) -> schemas.ModelMetadataResponse:
    """Endpoint lấy chi tiết metadata."""
    client_models_path = get_client_models_path(client_id)
    latest_metadata_file = _get_latest_metadata_file(client_models_path)

    if not latest_metadata_file:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy metadata cho client '{client_id}'"
        )

    try:
        with open(latest_metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Filter sensitive information (optional)
        filtered_metadata = {
            k: v for k, v in metadata.items()
            if k not in ['model_params', 'file_paths']  # Remove sensitive keys
        }

        return schemas.ModelMetadataResponse(
            client_id=client_id,
            metadata=filtered_metadata
        )

    except Exception as e:
        logger.error(f"Lỗi khi đọc metadata cho client {client_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi đọc metadata: {e}"
        )