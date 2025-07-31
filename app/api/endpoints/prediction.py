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
        client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
        model_name: str = FastApiPath(..., description="Model name: IN or OUT", example="IN")
):
    """Kiểm tra model có tồn tại không."""
    models_path = get_client_models_path(client_id)

    # Check for required files với model_name
    required_ht_preprocessor = models_path / f"preprocessor_hachtoan_{model_name}.joblib"
    required_ht_model = models_path / f"hachtoan_model_{model_name}.joblib"
    required_ht_encoder = models_path / "label_encoders" / f"hachtoan_encoder_{model_name}.joblib"

    if not all([required_ht_preprocessor.exists(), required_ht_model.exists(), required_ht_encoder.exists()]):
        logger.warning(f"Client {client_id}, Model {model_name}: Missing required HachToan model files.")
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found for client '{client_id}'. Please train the model first."
        )

    logger.debug(f"Client {client_id}, Model {model_name}: Model files verified.")

# --- API Endpoints ---

# --- Endpoint Gốc (Dự đoán kết hợp) ---
@router.post(
    "/{client_id}", # Giữ nguyên đường dẫn gốc
    response_model=schemas.PredictionResponse,
    summary="Dự đoán kết hợp HachToan và MaHangHoa",
    description="Nhận input cơ bản, dự đoán HachToan, sau đó dự đoán MaHangHoa nếu phù hợp.",
    dependencies=[Depends(check_hachtoan_model)],
    tags=["Prediction"], # Thêm tag để nhóm API
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Không tìm thấy mô hình HachToan cho client"},
        422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input"},
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server trong quá trình dự đoán"},
    }
)
async def predict_combined_endpoint(
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
    request_body: schemas.PredictionRequest = Body(...)
) -> schemas.PredictionResponse:
    """Endpoint dự đoán kết hợp HachToan -> MaHangHoa."""
    logger.info(f"Client {client_id}: Nhận yêu cầu dự đoán KẾT HỢP cho {len(request_body.items)} bản ghi.")
    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi chuẩn bị dữ liệu (kết hợp): {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý dữ liệu input: {e}")

    if input_df.empty and request_body.items:
         logger.warning(f"Client {client_id}: DataFrame rỗng sau khi chuẩn bị dữ liệu.")
         # Trả về lỗi hoặc response rỗng tùy logic
         raise HTTPException(status_code=400, detail="Dữ liệu input không hợp lệ hoặc rỗng.")

    try:
        prediction_results: List[dict] = predict_combined(client_id, input_df)

        if len(prediction_results) != len(request_body.items):
             logger.error(f"Client {client_id}: Số lượng kết quả (kết hợp) không khớp input.")
             raise HTTPException(status_code=500, detail="Lỗi nội bộ: Số lượng kết quả dự đoán không khớp.")

        response_items = [schemas.PredictionResultItem(**result) for result in prediction_results]

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi khi gọi predict_combined: {e}", exc_info=True)
        error_result = schemas.PredictionResultItem(is_outlier_input1=False, is_outlier_input2=False, error=f"Lỗi dự đoán nội bộ: {e}")
        # Cần tạo đủ số lượng item lỗi
        error_items = [schemas.PredictionResultItem(is_outlier_input1=False, is_outlier_input2=False, error=f"Lỗi dự đoán nội bộ: {e}") for _ in range(len(request_body.items))]
        return schemas.PredictionResponse(results=error_items)


    logger.info(f"Client {client_id}: Dự đoán kết hợp hoàn tất.")
    return schemas.PredictionResponse(results=response_items)


# --- Endpoint MỚI: Chỉ dự đoán HachToan ---
@router.post(
    "/{client_id}/hachtoan",
    response_model=schemas.HachToanPredictionResponse,
    summary="Chỉ dự đoán HachToan",
    description="Nhận input cơ bản và chỉ trả về dự đoán HachToan.",
    dependencies=[Depends(check_hachtoan_model)],
    tags=["Prediction"], # Thêm tag
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Không tìm thấy mô hình HachToan cho client"},
        422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input"},
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server trong quá trình dự đoán"},
    }
)
async def predict_hachtoan_endpoint(
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
    request_body: schemas.PredictionRequest = Body(...)
) -> schemas.HachToanPredictionResponse:
    """Endpoint chỉ dự đoán HachToan."""
    logger.info(f"Client {client_id}: Nhận yêu cầu dự đoán CHỈ HACHTOAN cho {len(request_body.items)} bản ghi.")
    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi chuẩn bị dữ liệu (chỉ HT): {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý dữ liệu input: {e}")

    if input_df.empty and request_body.items:
         logger.warning(f"Client {client_id}: DataFrame rỗng sau khi chuẩn bị dữ liệu (chỉ HT).")
         raise HTTPException(status_code=400, detail="Dữ liệu input không hợp lệ hoặc rỗng.")

    try:
        prediction_results: List[dict] = predict_hachtoan_only(client_id, input_df)

        if len(prediction_results) != len(request_body.items):
             logger.error(f"Client {client_id}: Số lượng kết quả (chỉ HT) không khớp input.")
             raise HTTPException(status_code=500, detail="Lỗi nội bộ: Số lượng kết quả dự đoán không khớp.")

        response_items = [schemas.HachToanPredictionResultItem(**result) for result in prediction_results]

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi khi gọi predict_hachtoan_only: {e}", exc_info=True)
        error_items = [schemas.HachToanPredictionResultItem(is_outlier_input1=False, error=f"Lỗi dự đoán nội bộ: {e}") for _ in range(len(request_body.items))]
        return schemas.HachToanPredictionResponse(results=error_items)

    logger.info(f"Client {client_id}: Dự đoán chỉ HachToan hoàn tất.")
    return schemas.HachToanPredictionResponse(results=response_items)


# --- Endpoint MỚI: Chỉ dự đoán MaHangHoa ---
@router.post(
    "/{client_id}/mahanghoa",
    response_model=schemas.MaHangHoaPredictionResponse,
    summary="Chỉ dự đoán MaHangHoa (cần cung cấp HachToan)",
    description="Nhận input cơ bản CÙNG VỚI HachToan đã biết và chỉ trả về dự đoán MaHangHoa.",
    dependencies=[Depends(check_mahanghoa_model)],
    tags=["Prediction"], # Thêm tag
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Không tìm thấy mô hình MaHangHoa cho client"},
        422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input (thiếu HachToan?)"},
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server trong quá trình dự đoán"},
    }
)
async def predict_mahanghoa_endpoint(
    client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
    request_body: schemas.MaHangHoaPredictionRequest = Body(...) # Dùng request schema MỚI
) -> schemas.MaHangHoaPredictionResponse:
    """Endpoint chỉ dự đoán MaHangHoa khi đã biết HachToan."""
    logger.info(f"Client {client_id}: Nhận yêu cầu dự đoán CHỈ MAHANGHOA cho {len(request_body.items)} bản ghi.")
    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi chuẩn bị dữ liệu (chỉ MH): {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý dữ liệu input: {e}")

    if input_df.empty and request_body.items:
         logger.warning(f"Client {client_id}: DataFrame rỗng sau khi chuẩn bị dữ liệu (chỉ MH).")
         raise HTTPException(status_code=400, detail="Dữ liệu input không hợp lệ hoặc rỗng.")

    # Kiểm tra lại xem HachToan có thực sự tồn tại trong df không sau prepare
    if config.TARGET_HACHTOAN not in input_df.columns:
        logger.error(f"Client {client_id}: Dữ liệu sau chuẩn bị thiếu cột '{config.TARGET_HACHTOAN}' cho dự đoán MaHangHoa.")
        raise HTTPException(status_code=400, detail=f"Dữ liệu input thiếu cột bắt buộc: {config.TARGET_HACHTOAN}")


    try:
        prediction_results: List[dict] = predict_mahanghoa_only(client_id, input_df)

        if len(prediction_results) != len(request_body.items):
             logger.error(f"Client {client_id}: Số lượng kết quả (chỉ MH) không khớp input.")
             raise HTTPException(status_code=500, detail="Lỗi nội bộ: Số lượng kết quả dự đoán không khớp.")

        response_items = [schemas.MaHangHoaPredictionResultItem(**result) for result in prediction_results]

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi khi gọi predict_mahanghoa_only: {e}", exc_info=True)
        error_items = [schemas.MaHangHoaPredictionResultItem(is_outlier_input2=False, error=f"Lỗi dự đoán nội bộ: {e}") for _ in range(len(request_body.items))]
        return schemas.MaHangHoaPredictionResponse(results=error_items)

    logger.info(f"Client {client_id}: Dự đoán chỉ MaHangHoa hoàn tất.")
    return schemas.MaHangHoaPredictionResponse(results=response_items)


# Endpoint prediction mới với model_name support
@router.post(
    "/{client_id}/{model_name}",
    response_model=schemas.PredictionResponse,
    summary="Predict với model name support",
    description="Dự đoán HachToan và MaHangHoa với model cụ thể (IN/OUT).",
    dependencies=[Depends(check_model_exists)],
    tags=["Prediction"],
    responses={
        404: {"model": schemas.ErrorResponse, "description": "Model không tồn tại cho client"},
        422: {"model": schemas.ErrorResponse, "description": "Lỗi validation dữ liệu input"},
        500: {"model": schemas.ErrorResponse, "description": "Lỗi server trong quá trình dự đoán"},
    }
)
async def predict_with_model_name(
        client_id: str = FastApiPath(..., description="ID của khách hàng", example="client_abc"),
        model_name: str = FastApiPath(..., description="Model name: IN or OUT", example="IN"),
        request_body: schemas.PredictionRequest = Body(...)
) -> schemas.PredictionResponse:
    """Endpoint dự đoán với model name support."""

    # Validate model_name
    if model_name not in config.VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name '{model_name}'. Must be one of: {config.VALID_MODEL_NAMES}"
        )

    logger.info(f"Client {client_id}, Model {model_name}: Prediction request for {len(request_body.items)} records.")

    try:
        input_list_of_dicts = [item.to_flat_dict() for item in request_body.items]
        input_df = prepare_prediction_data(input_list_of_dicts)
    except Exception as e:
        logger.error(f"Client {client_id}, Model {model_name}: Error preparing data: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error processing input data: {e}")

    if input_df.empty and request_body.items:
        logger.warning(f"Client {client_id}, Model {model_name}: DataFrame empty after data preparation.")
        raise HTTPException(status_code=400, detail="Input data is invalid or empty.")

    try:
        # Call predict_combined với model name support
        prediction_results: List[dict] = predict_combined(client_id, input_df)

        if len(prediction_results) != len(request_body.items):
            logger.error(f"Client {client_id}, Model {model_name}: Result count mismatch.")
            raise HTTPException(status_code=500, detail="Internal error: Prediction result count mismatch.")

        response_items = [schemas.PredictionResultItem(**result) for result in prediction_results]

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Client {client_id}, Model {model_name}: Error during prediction: {e}", exc_info=True)
        error_items = [
            schemas.PredictionResultItem(
                is_outlier_input1=False,
                is_outlier_input2=False,
                error=f"Prediction error: {e}"
            ) for _ in range(len(request_body.items))
        ]
        return schemas.PredictionResponse(results=error_items)

    logger.info(f"Client {client_id}, Model {model_name}: Prediction completed successfully.")
    return schemas.PredictionResponse(results=response_items)