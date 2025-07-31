# app/api/endpoints/prediction.py

import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Body, Path as FastApiPath
from typing import List

# --- Project Imports ---
from app.api import schemas # Import các Pydantic models đã cập nhật
import app.core.config as config # Import config module
# Import các hàm dự đoán đã refactor
from app.ml.models import predict_combined, predict_hachtoan_only, predict_mahanghoa_only
from app.ml.data_handler import prepare_prediction_data
from app.ml.utils import get_client_models_path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependencies kiểm tra model ---
async def check_hachtoan_model(
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc")
):
    """Kiểm tra sự tồn tại của preprocessor và encoder HachToan."""
    models_path = get_client_models_path(client_id)
    required_ht_preprocessor = models_path / "preprocessor_hachtoan.joblib"
    required_ht_encoder = models_path / "label_encoders" / config.HACHTOAN_ENCODER_FILENAME
    if not required_ht_preprocessor.exists() or not required_ht_encoder.exists():
        logger.warning(f"Client {client_id}: Thiếu các file model HachToan cần thiết.")
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy mô hình HachToan đã huấn luyện cho client '{client_id}'."
        )
    logger.debug(f"Client {client_id}: Đã tìm thấy thành phần model HachToan.")

async def check_mahanghoa_model(
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc")
):
    """Kiểm tra sự tồn tại của preprocessor và encoder MaHangHoa."""
    models_path = get_client_models_path(client_id)
    required_mh_preprocessor = models_path / "preprocessor_mahanghoa.joblib"
    required_mh_encoder = models_path / "label_encoders" / config.MAHANGHOA_ENCODER_FILENAME
    if not required_mh_preprocessor.exists() or not required_mh_encoder.exists():
        logger.warning(f"Client {client_id}: Thiếu các file model MaHangHoa cần thiết.")
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy mô hình MaHangHoa đã huấn luyện cho client '{client_id}'."
        )
    logger.debug(f"Client {client_id}: Đã tìm thấy thành phần model MaHangHoa.")


async def check_model_exists(
        tax_code: str = FastApiPath(..., description="Mã số thuế", example="0123456789"),
        model_type: str = FastApiPath(..., description="Loại model: IN hoặc OUT", example="IN")
):
    """Kiểm tra model có tồn tại cho tax_code và model_type"""

    # Validate model_type
    if model_type not in config.VALID_MODEL_NAMES:  # ["IN", "OUT"]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type '{model_type}'. Must be one of: {config.VALID_MODEL_NAMES}"
        )

    # Kiểm tra thư mục tax_code
    models_path = get_client_models_path(tax_code)  # /models/client_models/{tax_code}/

    # Kiểm tra các file model cần thiết
    required_files = [
        models_path / config.get_hachtoan_model_filename(model_type),  # hachtoan_model_IN.joblib
        models_path / config.get_preprocessor_hachtoan_filename(model_type),  # preprocessor_hachtoan_IN.joblib
        models_path / "label_encoders" / config.get_hachtoan_encoder_filename(model_type)  # hachtoan_encoder_IN.joblib
    ]

    missing_files = [f.name for f in required_files if not f.exists()]

    if missing_files:
        logger.warning(f"Tax code {tax_code}, Model {model_type}: Missing files: {missing_files}")
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_type}' not found for tax code '{tax_code}'. Missing files: {missing_files}"
        )

    logger.debug(f"Tax code {tax_code}, Model {model_type}: All required files found.")

# --- API Endpoints ---

# --- ENDPOINT CHÍNH: Combined Prediction ---
@router.post(
    "/{tax_code}/{model_type}",
    response_model=schemas.PredictionResponse,
    summary="Dự đoán HachToan và MaHangHoa",
    description="Dự đoán HachToan trước, sau đó dự đoán MaHangHoa dựa trên kết quả HachToan.",
    dependencies=[Depends(check_model_exists)],
    tags=["Prediction"],
    responses={
        400: {"model": schemas.ErrorResponse, "description": "Model type không hợp lệ"},
        404: {"model": schemas.ErrorResponse, "description": "Không tìm thấy model cho tax code và model type"},
        422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input"},
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server trong quá trình dự đoán"},
    }
)
async def predict_combined_endpoint(
        tax_code: str = FastApiPath(..., description="Mã số thuế", example="0123456789"),
        model_type: str = FastApiPath(..., description="Loại model: IN hoặc OUT", example="IN"),
        request_body: schemas.PredictionRequest = Body(...)
) -> schemas.PredictionResponse:
    """Endpoint dự đoán kết hợp HachToan -> MaHangHoa."""

    logger.info(f"Tax code {tax_code}, Model {model_type}: Prediction request for {len(request_body.items)} records.")

    try:
        # Chuẩn bị dữ liệu input
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)

        if input_df.empty and request_body.items:
            raise HTTPException(status_code=400, detail="Dữ liệu input không hợp lệ hoặc rỗng.")

        # Gọi prediction function với tax_code và model_type
        prediction_results: List[dict] = predict_combined(tax_code, model_type, input_df)

        if len(prediction_results) != len(request_body.items):
            raise HTTPException(status_code=500, detail="Lỗi nội bộ: Số lượng kết quả dự đoán không khớp.")

        # Chuyển đổi sang response schema
        response_items = [schemas.PredictionResultItem(**result) for result in prediction_results]

        logger.info(f"Tax code {tax_code}, Model {model_type}: Prediction completed successfully.")
        return schemas.PredictionResponse(results=response_items)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tax code {tax_code}, Model {model_type}: Prediction error: {e}", exc_info=True)
        # Trả về error response cho tất cả items
        error_items = [
            schemas.PredictionResultItem(
                is_outlier_input1=False,
                is_outlier_input2=False,
                error=f"Lỗi dự đoán: {e}"
            ) for _ in range(len(request_body.items))
        ]
        return schemas.PredictionResponse(results=error_items)


# --- ENDPOINT: Chỉ dự đoán HachToan ---
@router.post(
    "/{tax_code}/{model_type}/hachtoan",
    response_model=schemas.HachToanPredictionResponse,
    summary="Chỉ dự đoán HachToan",
    description="Chỉ dự đoán HachToan, không dự đoán MaHangHoa.",
    dependencies=[Depends(check_model_exists)],
    tags=["Prediction"]
)
async def predict_hachtoan_endpoint(
        tax_code: str = FastApiPath(..., description="Mã số thuế", example="0123456789"),
        model_type: str = FastApiPath(..., description="Loại model: IN hoặc OUT", example="IN"),
        request_body: schemas.PredictionRequest = Body(...)
) -> schemas.HachToanPredictionResponse:
    """Endpoint chỉ dự đoán HachToan."""

    logger.info(
        f"Tax code {tax_code}, Model {model_type}: HachToan-only prediction for {len(request_body.items)} records.")

    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)

        if input_df.empty and request_body.items:
            raise HTTPException(status_code=400, detail="Dữ liệu input không hợp lệ hoặc rỗng.")

        # Gọi prediction function
        prediction_results: List[dict] = predict_hachtoan_only(tax_code, model_type, input_df)

        if len(prediction_results) != len(request_body.items):
            raise HTTPException(status_code=500, detail="Lỗi nội bộ: Số lượng kết quả dự đoán không khớp.")

        response_items = [schemas.HachToanPredictionResultItem(**result) for result in prediction_results]

        logger.info(f"Tax code {tax_code}, Model {model_type}: HachToan prediction completed.")
        return schemas.HachToanPredictionResponse(results=response_items)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tax code {tax_code}, Model {model_type}: HachToan prediction error: {e}", exc_info=True)
        error_items = [
            schemas.HachToanPredictionResultItem(
                is_outlier_input1=False,
                error=f"Lỗi dự đoán HachToan: {e}"
            ) for _ in range(len(request_body.items))
        ]
        return schemas.HachToanPredictionResponse(results=error_items)


# --- ENDPOINT: Chỉ dự đoán MaHangHoa ---
@router.post(
    "/{tax_code}/{model_type}/mahanghoa",
    response_model=schemas.MaHangHoaPredictionResponse,
    summary="Chỉ dự đoán MaHangHoa (cần HachToan input)",
    description="Chỉ dự đoán MaHangHoa khi đã biết HachToan.",
    dependencies=[Depends(check_model_exists)],
    tags=["Prediction"]
)
async def predict_mahanghoa_endpoint(
        tax_code: str = FastApiPath(..., description="Mã số thuế", example="0123456789"),
        model_type: str = FastApiPath(..., description="Loại model: IN hoặc OUT", example="IN"),
        request_body: schemas.MaHangHoaPredictionRequest = Body(...)
) -> schemas.MaHangHoaPredictionResponse:
    """Endpoint chỉ dự đoán MaHangHoa."""

    logger.info(
        f"Tax code {tax_code}, Model {model_type}: MaHangHoa-only prediction for {len(request_body.items)} records.")

    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)

        if input_df.empty and request_body.items:
            raise HTTPException(status_code=400, detail="Dữ liệu input không hợp lệ hoặc rỗng.")

        # Kiểm tra HachToan input
        if config.TARGET_HACHTOAN not in input_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Dữ liệu input thiếu cột bắt buộc: {config.TARGET_HACHTOAN}"
            )

        # Gọi prediction function
        prediction_results: List[dict] = predict_mahanghoa_only(tax_code, model_type, input_df)

        if len(prediction_results) != len(request_body.items):
            raise HTTPException(status_code=500, detail="Lỗi nội bộ: Số lượng kết quả dự đoán không khớp.")

        response_items = [schemas.MaHangHoaPredictionResultItem(**result) for result in prediction_results]

        logger.info(f"Tax code {tax_code}, Model {model_type}: MaHangHoa prediction completed.")
        return schemas.MaHangHoaPredictionResponse(results=response_items)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tax code {tax_code}, Model {model_type}: MaHangHoa prediction error: {e}", exc_info=True)
        error_items = [
            schemas.MaHangHoaPredictionResultItem(
                is_outlier_input2=False,
                error=f"Lỗi dự đoán MaHangHoa: {e}"
            ) for _ in range(len(request_body.items))
        ]
        return schemas.MaHangHoaPredictionResponse(results=error_items)


# # Endpoint prediction mới với model_name support
# @router.post(
#     "/{client_id}/{model_name}",
#     response_model=schemas.PredictionResponse,
#     summary="Predict với model name support",
#     description="Dự đoán HachToan và MaHangHoa với model cụ thể (IN/OUT).",
#     dependencies=[Depends(check_model_exists)],
#     tags=["Prediction"],
#     responses={
#         404: {"model": schemas.ErrorResponse, "description": "Model không tồn tại cho client"},
#         422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input"},
#         500: {"model": schemas.ErrorResponse, "description": "Lỗi server trong quá trình dự đoán"},
#     }
# )
# async def predict_with_model_name(
#         client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
#         model_name: str = FastApiPath(..., description="Model name: IN or OUT", example="IN"),
#         request_body: schemas.PredictionRequest = Body(...)
# ) -> schemas.PredictionResponse:
#     """Endpoint dự đoán với model name support."""
#
#     # Validate model_name
#     if model_name not in config.VALID_MODEL_NAMES:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid model_name '{model_name}'. Must be one of: {config.VALID_MODEL_NAMES}"
#         )
#
#     logger.info(f"Client {client_id}, Model {model_name}: Prediction request for {len(request_body.items)} records.")
#
#     try:
#         input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
#         input_df = prepare_prediction_data(input_list_of_dicts)
#     except Exception as e:
#         logger.error(f"Client {client_id}, Model {model_name}: Error preparing data: {e}", exc_info=True)
#         raise HTTPException(status_code=400, detail=f"Error processing input data: {e}")
#
#     if input_df.empty and request_body.items:
#         logger.warning(f"Client {client_id}, Model {model_name}: DataFrame empty after data preparation.")
#         raise HTTPException(status_code=400, detail="Input data is invalid or empty.")
#
#     try:
#         # Call predict_combined với model name support
#         prediction_results: List[dict] = predict_combined(client_id, input_df)
#
#         if len(prediction_results) != len(request_body.items):
#             logger.error(f"Client {client_id}, Model {model_name}: Result count mismatch.")
#             raise HTTPException(status_code=500, detail="Internal error: Prediction result count mismatch.")
#
#         response_items = [schemas.PredictionResultItem(**result) for result in prediction_results]
#
#     except HTTPException as http_exc:
#         raise http_exc
#     except Exception as e:
#         logger.error(f"Client {client_id}, Model {model_name}: Error during prediction: {e}", exc_info=True)
#         error_items = [
#             schemas.PredictionResultItem(
#                 is_outlier_input1=False,
#                 is_outlier_input2=False,
#                 error=f"Prediction error: {e}"
#             ) for _ in range(len(request_body.items))
#         ]
#         return schemas.PredictionResponse(results=error_items)
#
#     logger.info(f"Client {client_id}, Model {model_name}: Prediction completed successfully.")
#     return schemas.PredictionResponse(results=response_items)