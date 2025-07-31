# app/ml/models.py

import pandas as pd
import numpy as np
import logging
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable
import os

import app.core.config as config
from app.ml.data_handler import load_all_client_data

# Import các lớp model cần hỗ trợ
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
# Thêm import cho các model khác nếu cần (ví dụ: lightgbm)
# try:
#     import lightgbm as lgb
# except ImportError:
#     lgb = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

import app.core.config as config
from app.ml.utils import (
    get_client_models_path, get_client_label_encoder_path,
    save_joblib, load_joblib,
)
from app.ml.data_handler import load_all_client_data
from app.ml.pipeline import (
    create_hachtoan_preprocessor, create_mahanghoa_preprocessor
)
from app.ml.outlier_detector import (
    train_outlier_detector, load_outlier_detector, check_outlier
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions

def _filter_data_by_model_type(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Lọc dữ liệu theo model type (IN/OUT) dựa trên ký tự đầu của HachToan.

    Args:
        df: DataFrame chứa dữ liệu training
        model_name: "IN" hoặc "OUT"

    Returns:
        DataFrame đã được lọc
    """
    if config.TARGET_HACHTOAN not in df.columns:
        logger.warning(f"Cột {config.TARGET_HACHTOAN} không tồn tại trong DataFrame")
        return df

    # Đảm bảo HachToan là string và loại bỏ NaN
    df_filtered = df.copy()
    df_filtered = df_filtered.dropna(subset=[config.TARGET_HACHTOAN])
    df_filtered[config.TARGET_HACHTOAN] = df_filtered[config.TARGET_HACHTOAN].astype(str)

    original_count = len(df_filtered)

    if model_name == "IN":
        # Model IN: Loại bỏ các bản ghi bắt đầu bằng '5' hoặc '7'
        df_filtered = df_filtered[
            ~df_filtered[config.TARGET_HACHTOAN].str.startswith(('5', '7'))
        ]
        logger.info(f"Model IN: Loại bỏ records có HachToan bắt đầu bằng '5' hoặc '7'. "
                    f"From {original_count} to {len(df_filtered)} records")

    elif model_name == "OUT":
        # Model OUT: Chỉ giữ lại các bản ghi bắt đầu bằng '5' hoặc '7'
        df_filtered = df_filtered[
            df_filtered[config.TARGET_HACHTOAN].str.startswith(('5', '7'))
        ]
        logger.info(f"Model OUT: Chỉ giữ records có HachToan bắt đầu bằng '5' hoặc '7'. "
                    f"From {original_count} to {len(df_filtered)} records")
    else:
        logger.warning(f"Unknown model_name: {model_name}. No filtering applied.")

    if len(df_filtered) == 0:
        logger.error(f"Không còn dữ liệu nào sau khi lọc cho model {model_name}!")

    return df_filtered
def _find_available_model(client_id: str) -> Optional[str]:
    """Tìm model_name có sẵn cho client (IN hoặc OUT)"""
    models_path = get_client_models_path(client_id)

    for model_name in config.VALID_MODEL_NAMES:  # ["IN", "OUT"]
        # Kiểm tra file chính cần thiết
        hachtoan_model_file = config.get_hachtoan_model_filename(model_name)
        preprocessor_file = config.get_preprocessor_hachtoan_filename(model_name)
        encoder_file = config.get_hachtoan_encoder_filename(model_name)

        if all([
            (models_path / hachtoan_model_file).exists(),
            (models_path / preprocessor_file).exists(),
            (models_path / "label_encoders" / encoder_file).exists()
        ]):
            logger.info(f"Found available model '{model_name}' for client {client_id}")
            return model_name

    # Fallback: kiểm tra format cũ
    old_files = [
        models_path / "hachtoan_model.joblib",
        models_path / "preprocessor_hachtoan.joblib",
        models_path / "label_encoders" / "hachtoan_encoder.joblib"
    ]

    if all(f.exists() for f in old_files):
        logger.info(f"Found legacy model format for client {client_id}")
        return "legacy"

    logger.error(f"No trained models found for client {client_id}")
    return None


def _load_model_components(client_id: str, model_name: str, target: str = "hachtoan"):
    """
    Load model components với model_name hoặc legacy format
    target: "hachtoan" hoặc "mahanghoa"
    """
    models_path = get_client_models_path(client_id)

    if model_name == "legacy":
        # Load old format
        if target == "hachtoan":
            preprocessor_file = "preprocessor_hachtoan.joblib"
            model_file = "hachtoan_model.joblib"
            encoder_file = "hachtoan_encoder.joblib"
            outlier_file = "outlier_detector_1.joblib"
        else:  # mahanghoa
            preprocessor_file = "preprocessor_mahanghoa.joblib"
            model_file = "mahanghoa_model.joblib"
            encoder_file = "mahanghoa_encoder.joblib"
            outlier_file = "outlier_detector_2.joblib"
    else:
        # Load new format với model_name
        if target == "hachtoan":
            preprocessor_file = config.get_preprocessor_hachtoan_filename(model_name)
            model_file = config.get_hachtoan_model_filename(model_name)
            encoder_file = config.get_hachtoan_encoder_filename(model_name)
            outlier_file = config.get_outlier_detector_1_filename(model_name)
        else:  # mahanghoa
            preprocessor_file = config.get_preprocessor_mahanghoa_filename(model_name)
            model_file = config.get_mahanghoa_model_filename(model_name)
            encoder_file = config.get_mahanghoa_encoder_filename(model_name)
            outlier_file = config.get_outlier_detector_2_filename(model_name)

    # Load components
    preprocessor = load_joblib(models_path / preprocessor_file)
    model = load_joblib(models_path / model_file)
    encoder = _load_label_encoder(client_id, encoder_file)
    outlier_detector = load_outlier_detector(client_id, outlier_file)

    return preprocessor, model, encoder, outlier_detector

def _find_latest_metadata_for_model(models_path: Path, model_name: str) -> Optional[Path]:
    """Tìm file metadata mới nhất cho một model_name cụ thể."""
    metadata_pattern = f"{config.METADATA_FILENAME_PREFIX}{model_name}_*.json"
    metadata_files = sorted(
        models_path.glob(metadata_pattern),
        key=os.path.getmtime,
        reverse=True
    )
    if metadata_files:
        return metadata_files[0]
    return None

def make_json_serializable(obj):
    # ... (code giữ nguyên) ...
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, (datetime, Path)): return str(obj)
    elif isinstance(obj, dict): return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [make_json_serializable(i) for i in obj]
    return obj

def _fit_or_load_label_encoder(client_id: str, target_series: pd.Series, encoder_filename: str) -> Optional[LabelEncoder]:
    # ... (code giữ nguyên) ...
    encoder_path = get_client_label_encoder_path(client_id) / encoder_filename
    if target_series.empty:
        if encoder_path.exists():
            try: encoder_path.unlink(); logger.info(f"Đã xóa LabelEncoder cũ: {encoder_path}")
            except OSError as e: logger.error(f"Lỗi khi xóa LabelEncoder cũ {encoder_path}: {e}")
        return None
    logger.info(f"Fit lại LabelEncoder cho {target_series.name}...")
    try:
        label_encoder = LabelEncoder(); label_encoder.fit(target_series.astype(str))
        save_joblib(label_encoder, encoder_path)
        return label_encoder
    except Exception as e:
        logger.error(f"Lỗi khi fit/lưu LabelEncoder cho {target_series.name}: {e}", exc_info=True)
        if encoder_path.exists():
            try: encoder_path.unlink()
            except OSError: pass
        return None

def _load_label_encoder(client_id: str, encoder_filename: str) -> Optional[LabelEncoder]:
    encoder_path = get_client_label_encoder_path(client_id) / encoder_filename
    return load_joblib(encoder_path)

# --- Core Training Logic (Cập nhật để nhận model object) ---
def train_single_model(
    client_id: str,
    df: pd.DataFrame,
    target_column: str,
    preprocessor_creator: Callable[[List[str]], ColumnTransformer],
    preprocessor_filename: str,
    # Thay vì model_filename, nhận model object đã khởi tạo
    model_object: Any, # Model đã được khởi tạo từ train_client_models
    model_save_filename: str, # Tên file để lưu model
    encoder_filename: str,
    outlier_detector_filename: str,
    validation_size: float = config.VALIDATION_SET_SIZE # Lấy từ config
) -> Tuple[Optional[ColumnTransformer], Optional[Any], Optional[LabelEncoder], Optional[Dict[str, Any]]]:
    """
    Huấn luyện một model object cụ thể, outlier detector, đánh giá, và trả về metrics.
    """
    models_path = get_client_models_path(client_id)
    model_path = models_path / model_save_filename # Dùng tên file được truyền vào
    preprocessor_path = models_path / preprocessor_filename
    encoder_path = get_client_label_encoder_path(client_id) / encoder_filename
    outlier_path = models_path / outlier_detector_filename

    # --- Kiểm tra cột target và dữ liệu rỗng (Giữ nguyên) ---
    if target_column not in df.columns:
        logger.warning(f"Cột target '{target_column}' không tồn tại. Bỏ qua huấn luyện.")
        if model_path.exists(): model_path.unlink(); logger.info(f"Đã xóa model cũ: {model_path}")
        # ... (Xóa các file khác) ...
        return None, None, None, None
    if df.empty:
        logger.warning(f"Không có dữ liệu để huấn luyện target '{target_column}'.")
        if model_path.exists(): model_path.unlink(); logger.info(f"Đã xóa model cũ: {model_path}")
        # ... (Xóa các file khác) ...
        return None, None, None, None
    # --------------------------------------

    logger.info(f"Bắt đầu huấn luyện model '{type(model_object).__name__}' cho target: '{target_column}' với {len(df)} bản ghi.")

    # --- Tách X, y (Giữ nguyên) ---
    input_cols_for_model = [col for col in config.INPUT_COLUMNS if col in df.columns]
    if target_column == config.TARGET_MAHANGHOA and config.TARGET_HACHTOAN in df.columns:
        if config.TARGET_HACHTOAN not in input_cols_for_model:
            input_cols_for_model.append(config.TARGET_HACHTOAN)
    if not input_cols_for_model:
         logger.error(f"Không tìm thấy cột input nào hợp lệ để huấn luyện {target_column}.")
         return None, None, None, None
    try: X = df[input_cols_for_model]; y = df[target_column]
    except KeyError as e:
        logger.error(f"Lỗi KeyError khi tách X, y cho target {target_column}: {e}.")
        return None, None, None, None

    # --- Fit/Load Label Encoder (Giữ nguyên) ---
    label_encoder = _fit_or_load_label_encoder(client_id, y, encoder_filename)
    if label_encoder is None: return None, None, None, None
    try: y_encoded = label_encoder.transform(y.astype(str))
    except ValueError as e:
         logger.error(f"Lỗi khi transform target '{target_column}': {e}.")
         return None, None, label_encoder, None
    num_classes = len(label_encoder.classes_)
    logger.info(f"Số lớp (classes) trong target '{target_column}': {num_classes}")
    if num_classes == 0: return None, None, label_encoder, None

    # --- Tạo và Fit Preprocessor (Giữ nguyên) ---
    logger.info("Tạo và fit preprocessor...")
    actual_input_cols_for_preprocessor = list(X.columns)
    preprocessor = preprocessor_creator(actual_input_cols_for_preprocessor)
    try:
        preprocessor.fit(X); save_joblib(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor đã được fit và lưu vào: {preprocessor_path}")
    except Exception as e:
        logger.error(f"Lỗi khi fit hoặc lưu preprocessor: {e}", exc_info=True)
        if preprocessor_path.exists(): preprocessor_path.unlink()
        return None, None, label_encoder, None

    # --- Transform dữ liệu (Giữ nguyên) ---
    logger.info("Transforming toàn bộ dữ liệu với preprocessor...")
    try:
        X_transformed = preprocessor.transform(X)
        logger.info(f"Dữ liệu đã transform. Shape: {X_transformed.shape if hasattr(X_transformed, 'shape') else 'N/A'}")
    except Exception as e:
        logger.error(f"Lỗi khi transform dữ liệu: {e}", exc_info=True)
        return preprocessor, None, label_encoder, None

    # --- Huấn luyện Outlier Detector (Giữ nguyên) ---
    logger.info(f"Huấn luyện Outlier Detector ({outlier_detector_filename})...")
    _ = train_outlier_detector(client_id=client_id, data=X_transformed, detector_filename=outlier_detector_filename)

    # --- Đánh giá mô hình (Sử dụng model_object đã truyền vào) ---
    metrics = None
    final_model = model_object # Sử dụng model object được truyền vào làm model cuối cùng
    if num_classes >= 2 and validation_size > 0 and validation_size < 1:
        logger.info(f"Thực hiện đánh giá mô hình {type(final_model).__name__} trên {validation_size*100:.1f}% validation set...")
        try:
            X_train_eval, X_val, y_train_eval, y_val = train_test_split(
                X_transformed, y_encoded, test_size=validation_size, random_state=42, stratify=y_encoded
            )
            logger.info(f"Train set size (for eval): {X_train_eval.shape[0]}, Validation set size: {X_val.shape[0]}")

            # Huấn luyện bản sao của model trên tập train (để đánh giá)
            # Sử dụng clone để không làm thay đổi model gốc nếu fit thay đổi trạng thái nội bộ
            from sklearn.base import clone # Import clone
            eval_model = clone(final_model)
            eval_model.fit(X_train_eval, y_train_eval)
            y_pred_val = eval_model.predict(X_val)

            report = classification_report(
                y_val, y_pred_val, target_names=label_encoder.classes_,
                output_dict=True, zero_division=0
            )
            logger.info(f"Kết quả đánh giá (Validation Set) cho {target_column}:\n{json.dumps(report, indent=2)}")
            metrics = report

        except Exception as e:
            logger.error(f"Lỗi trong quá trình đánh giá mô hình {target_column}: {e}", exc_info=True)
            metrics = {"error": f"Evaluation failed: {e}"}

    # --- Huấn luyện mô hình cuối cùng trên TOÀN BỘ dữ liệu (Sử dụng final_model) ---
    logger.info(f"Huấn luyện mô hình cuối cùng {type(final_model).__name__} cho {target_column} trên toàn bộ {X_transformed.shape[0]} mẫu...")
    if num_classes < 2:
         logger.warning(f"Chỉ có {num_classes} lớp. Không huấn luyện mô hình phân loại cuối cùng.")
         if model_path.exists(): model_path.unlink()
         return preprocessor, None, label_encoder, metrics

    try:
        final_model.fit(X_transformed, y_encoded) # Fit model object đã truyền vào trên toàn bộ dữ liệu
        logger.info(f"Mô hình cuối cùng {type(final_model).__name__} đã huấn luyện xong.")
        save_joblib(final_model, model_path) # Lưu model object này
        logger.info(f"Mô hình cuối cùng đã được lưu vào: {model_path}")
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình cuối cùng: {e}", exc_info=True)
        if model_path.exists(): model_path.unlink()
        final_model = None # Đặt lại là None nếu lỗi

    return preprocessor, final_model, label_encoder, metrics

# --- Function to get latest metadata file ---
def _find_latest_metadata_file(models_path: Path) -> Optional[Path]:
    """Tìm file metadata mới nhất trong thư mục models."""
    metadata_files = sorted(
        models_path.glob(f"{config.METADATA_FILENAME_PREFIX}*.json"),
        key=os.path.getmtime, # Sắp xếp theo thời gian sửa đổi
        reverse=True
    )
    if metadata_files:
        return metadata_files[0]
    return None

# --- Function to load model type from metadata ---
def _load_model_type_from_metadata(metadata_file: Path) -> Optional[str]:
    """Đọc loại model đã lưu từ file metadata."""
    if not metadata_file or not metadata_file.exists():
        return None
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Tìm key lưu tên model (ví dụ: 'selected_model_type' hoặc trong 'hachtoan_model_info')
        # Cần đảm bảo key này được lưu nhất quán
        model_type = metadata.get("selected_model_type") # Ưu tiên key này nếu có
        if not model_type and "hachtoan_model_info" in metadata:
            model_type = metadata["hachtoan_model_info"].get("model_class") # Lấy từ info HachToan

        if model_type and isinstance(model_type, str):
             # Kiểm tra xem model_type có nằm trong danh sách hỗ trợ không
             if model_type in [e.value for e in config.SupportedModels]:
                 logger.info(f"Đã đọc model type '{model_type}' từ metadata: {metadata_file.name}")
                 return model_type
             else:
                  logger.warning(f"Model type '{model_type}' đọc từ metadata không được hỗ trợ.")
                  return None
        else:
             logger.warning(f"Không tìm thấy hoặc định dạng key model type không đúng trong metadata: {metadata_file.name}")
             return None

    except Exception as e:
        logger.error(f"Lỗi khi đọc model type từ metadata {metadata_file.name}: {e}")
        return None

# --- Function to instantiate model ---
def _instantiate_model(model_type_str: str) -> Optional[Any]:
    """Khởi tạo đối tượng model dựa trên tên và tham số mặc định từ config."""
    logger.info(f"Khởi tạo model loại: {model_type_str}")
    params = config.DEFAULT_MODEL_PARAMS.get(model_type_str, {})
    logger.debug(f"Sử dụng tham số mặc định: {params}")

    try:
        if model_type_str == config.SupportedModels.RANDOM_FOREST.value:
            return RandomForestClassifier(**params)
        elif model_type_str == config.SupportedModels.LOGISTIC_REGRESSION.value:
            return LogisticRegression(**params)
        elif model_type_str == config.SupportedModels.MULTINOMIAL_NB.value:
             # Kiểm tra điều kiện đặc biệt cho MNB nếu cần (ví dụ input phải >= 0)
            return MultinomialNB(**params)
        elif model_type_str == config.SupportedModels.LINEAR_SVC.value:
            return LinearSVC(**params)
        # Thêm các elif cho các model khác
        # elif model_type_str == config.SupportedModels.LIGHTGBM.value:
        #     if lgb: return lgb.LGBMClassifier(**params)
        #     else: logger.error("Thư viện LightGBM chưa được cài đặt."); return None
        else:
            logger.error(f"Loại model không xác định hoặc không được hỗ trợ: {model_type_str}")
            return None
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo model {model_type_str} với params {params}: {e}", exc_info=True)
        return None


# --- Main Training Orchestrator (Cập nhật) ---
def train_client_models(
        client_id: str,
        model_name: str,  # NEW PARAMETER
        initial_model_type_str: Optional[str] = None
) -> bool:
    """
    Enhanced train_client_models với model_name support và data filtering.

    Args:
        client_id: ID của client
        model_name: "IN" hoặc "OUT"
        initial_model_type_str: Loại model (chỉ cần cho lần đầu)
    """
    # Validate model_name
    if model_name not in config.VALID_MODEL_NAMES:
        logger.error(f"Invalid model_name: {model_name}. Must be one of {config.VALID_MODEL_NAMES}")
        return False

    start_time = time.time()
    training_timestamp_utc = datetime.now(timezone.utc)
    logger.info(f"===== Training model '{model_name}' for client: {client_id} at {training_timestamp_utc} UTC =====")

    models_path = get_client_models_path(client_id)

    # Initialize metadata với model_name
    metadata = {
        "client_id": client_id,
        "model_name": model_name,  # NEW FIELD
        "training_timestamp_utc": training_timestamp_utc.isoformat(),
        "status": "STARTED",
        "selected_model_type": None,
        "data_info": {},
        "hachtoan_model_info": {},
        "mahanghoa_model_info": {},
        "training_duration_seconds": None,
        "error_message": None
    }

    # --- Determine model type ---
    model_type_to_train = None

    if initial_model_type_str:
        # New training
        if initial_model_type_str in [e.value for e in config.SupportedModels]:
            model_type_to_train = initial_model_type_str
            metadata["selected_model_type"] = model_type_to_train
            logger.info(f"New training for model '{model_name}' with type: {model_type_to_train}")
        else:
            error_msg = f"Invalid model type: {initial_model_type_str}"
            logger.error(error_msg)
            metadata["status"] = "FAILED"
            metadata["error_message"] = error_msg
            return False
    else:
        # Retrain - find existing model type
        existing_metadata = _find_latest_metadata_for_model(models_path, model_name)
        if existing_metadata:
            saved_model_type = _load_model_type_from_metadata(existing_metadata)
            model_type_to_train = saved_model_type or config.DEFAULT_MODEL_TYPE.value
            metadata["selected_model_type"] = model_type_to_train
            logger.info(f"Retraining model '{model_name}' with existing type: {model_type_to_train}")
        else:
            # No existing model - use default
            model_type_to_train = config.DEFAULT_MODEL_TYPE.value
            metadata["selected_model_type"] = model_type_to_train
            logger.info(f"No existing model '{model_name}' found, using default type: {model_type_to_train}")

    # --- Load data from database ---
    from app.ml.database import load_all_training_data

    df_raw = load_all_training_data(client_id)
    if df_raw is None or df_raw.empty:
        error_msg = f"No training data found in database for client {client_id}"
        logger.error(error_msg)
        metadata["status"] = "FAILED"
        metadata["error_message"] = error_msg
        return False

    # *** THÊM BƯỚC LỌCC DỮ LIỆU THEO MODEL TYPE ***
    logger.info(f"Raw data loaded: {len(df_raw)} records")
    df = _filter_data_by_model_type(df_raw, model_name)

    if df.empty:
        error_msg = f"No data remaining after filtering for model {model_name}"
        logger.error(error_msg)
        metadata["status"] = "FAILED"
        metadata["error_message"] = error_msg
        metadata["data_info"]["raw_samples_loaded"] = len(df_raw)
        metadata["data_info"]["filtered_samples"] = len(df)
        metadata["data_info"]["filter_rule"] = f"Model {model_name}: " + (
            "Exclude HachToan starting with '5' or '7'" if model_name == "IN"
            else "Only HachToan starting with '5' or '7'"
        )
        return False

    metadata["data_info"]["raw_samples_loaded"] = len(df_raw)
    metadata["data_info"]["filtered_samples"] = len(df)
    metadata["data_info"]["columns_present"] = df.columns.tolist()
    metadata["data_info"]["filter_rule"] = f"Model {model_name}: " + (
        "Exclude HachToan starting with '5' or '7'" if model_name == "IN"
        else "Only HachToan starting with '5' or '7'"
    )

    logger.info(f"After filtering for model '{model_name}': {len(df)} records")

    # --- Tiếp tục với logic training như cũ ---
    # Create model instances
    model_ht_instance = _instantiate_model(model_type_to_train)
    model_mh_instance = _instantiate_model(model_type_to_train)

    if model_ht_instance is None:
        error_msg = f"Cannot instantiate model type: {model_type_to_train}"
        logger.error(error_msg)
        metadata["status"] = "FAILED"
        metadata["error_message"] = error_msg
        return False

    # --- Generate filenames with model_name ---
    preprocessor_hachtoan_file = config.get_preprocessor_hachtoan_filename(model_name)
    hachtoan_model_save_file = config.get_hachtoan_model_filename(model_name)
    mahanghoa_model_save_file = config.get_mahanghoa_model_filename(model_name)
    hachtoan_encoder_file = config.get_hachtoan_encoder_filename(model_name)
    mahanghoa_encoder_file = config.get_mahanghoa_encoder_filename(model_name)
    outlier_detector_1_file = config.get_outlier_detector_1_filename(model_name)
    outlier_detector_2_file = config.get_outlier_detector_2_filename(model_name)
    preprocessor_mahanghoa_file = config.get_preprocessor_mahanghoa_filename(model_name)

    # --- Train HachToan model ---
    logger.info(f"--- Training {model_type_to_train} for HachToan (model: {model_name}) ---")
    prep_ht, model_ht, enc_ht, metrics_ht = train_single_model(
        client_id=client_id,
        df=df,  # Sử dụng df đã được lọc
        target_column=config.TARGET_HACHTOAN,
        preprocessor_creator=create_hachtoan_preprocessor,
        preprocessor_filename=preprocessor_hachtoan_file,
        model_object=model_ht_instance,
        model_save_filename=hachtoan_model_save_file,
        encoder_filename=hachtoan_encoder_file,
        outlier_detector_filename=outlier_detector_1_file
    )

    # Save HachToan metadata
    metadata["hachtoan_model_info"]["preprocessor_saved"] = prep_ht is not None
    metadata["hachtoan_model_info"]["model_saved"] = model_ht is not None
    metadata["hachtoan_model_info"]["encoder_saved"] = enc_ht is not None
    if model_ht:
        metadata["hachtoan_model_info"]["model_class"] = type(model_ht).__name__
        metadata["hachtoan_model_info"]["model_params"] = model_ht.get_params()
    metadata["hachtoan_model_info"]["evaluation_metrics"] = metrics_ht

    if prep_ht is None or enc_ht is None:
        error_msg = "HachToan training failed (preprocessor/encoder)"
        logger.error(error_msg)
        metadata["status"] = "FAILED"
        metadata["error_message"] = error_msg
        return False

    # --- Train MaHangHoa model ---
    logger.info(f"--- Training {model_type_to_train} for MaHangHoa (model: {model_name}) ---")
    metadata["mahanghoa_model_info"]["attempted"] = False

    # Filter data for MaHangHoa (sử dụng df đã được lọc theo model type)
    if config.TARGET_MAHANGHOA in df.columns:
        df_filtered_prefix = df[
            df[config.TARGET_HACHTOAN].astype(str).str.startswith(tuple(config.HACHTOAN_PREFIX_FOR_MAHANGHOA))
        ].copy()

        if not df_filtered_prefix.empty:
            df_mahanghoa = df_filtered_prefix.dropna(subset=[config.TARGET_MAHANGHOA]).copy()
            df_mahanghoa = df_mahanghoa[df_mahanghoa[config.TARGET_MAHANGHOA].astype(str) != '']
        else:
            df_mahanghoa = pd.DataFrame()
    else:
        df_mahanghoa = pd.DataFrame()

    metadata["data_info"]["samples_for_mahanghoa"] = len(df_mahanghoa)
    logger.info(f"Data for MaHangHoa training after all filtering: {len(df_mahanghoa)} records")

    if not df_mahanghoa.empty:
        metadata["mahanghoa_model_info"]["attempted"] = True
        prep_mh, model_mh, enc_mh, metrics_mh = train_single_model(
            client_id=client_id,
            df=df_mahanghoa,
            target_column=config.TARGET_MAHANGHOA,
            preprocessor_creator=create_mahanghoa_preprocessor,
            preprocessor_filename=preprocessor_mahanghoa_file,
            model_object=model_mh_instance,
            model_save_filename=mahanghoa_model_save_file,
            encoder_filename=mahanghoa_encoder_file,
            outlier_detector_filename=outlier_detector_2_file
        )

        metadata["mahanghoa_model_info"]["preprocessor_saved"] = prep_mh is not None
        metadata["mahanghoa_model_info"]["model_saved"] = model_mh is not None
        metadata["mahanghoa_model_info"]["encoder_saved"] = enc_mh is not None
        metadata["mahanghoa_model_info"]["evaluation_metrics"] = metrics_mh
    else:
        logger.warning(f"No suitable data found for MaHangHoa training after filtering (model: {model_name})")
        metadata["mahanghoa_model_info"]["message"] = "No suitable data for training MaHangHoa after filtering"

    # --- Save metadata with model_name ---
    end_time = time.time()
    metadata["training_duration_seconds"] = round(end_time - start_time, 2)
    metadata["status"] = "COMPLETED"

    timestamp_str = training_timestamp_utc.strftime('%Y%m%d_%H%M%S')
    meta_filename = config.get_metadata_filename(model_name, timestamp_str)

    try:
        serializable_metadata = make_json_serializable(metadata)
        with open(models_path / meta_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_metadata, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved training metadata: {meta_filename}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")

    logger.info(
        f"===== Training completed for model '{model_name}' of client {client_id} ({metadata['training_duration_seconds']:.2f}s) =====")
    return True


# --- Prediction Logic (Refactored) ---
def _predict_hachtoan_batch(
    client_id: str,
    model_name: str,  # THÊM PARAMETER MỚI
    input_df: pd.DataFrame,
    preprocessor_ht: Optional[ColumnTransformer],
    model_ht: Optional[Any],
    encoder_ht: Optional[LabelEncoder],
    outlier_detector_1: Optional[Any]
) -> List[Dict[str, Any]]:
    """Dự đoán chỉ HachToan cho một batch DataFrame với model_name support."""
    results = []
    n_items = len(input_df)
    y_pred_ht = [None] * n_items
    probabilities_ht = [None] * n_items
    outlier_flags_1 = [False] * n_items
    errors = [None] * n_items

    if not preprocessor_ht or not encoder_ht:
        logger.error(f"Client {client_id}, Model {model_name}: Thiếu preprocessor hoặc encoder HachToan.")
        for i in range(n_items):
            results.append({"error": f"Thiếu thành phần model HachToan cho model {model_name}."})
        return results

    can_check_outlier_1 = bool(preprocessor_ht and outlier_detector_1)

    try:
        expected_features_ht = list(preprocessor_ht.feature_names_in_)
        input_data_aligned_ht = input_df.reindex(columns=expected_features_ht, fill_value="")
        X_transformed_ht = preprocessor_ht.transform(input_data_aligned_ht)

        if can_check_outlier_1:
            outlier_flags_1 = check_outlier(outlier_detector_1, X_transformed_ht)

        if model_ht:
            y_pred_encoded_ht = model_ht.predict(X_transformed_ht)
            y_pred_proba_ht = model_ht.predict_proba(X_transformed_ht)
            y_pred_ht = encoder_ht.inverse_transform(y_pred_encoded_ht)
            probabilities_ht = np.max(y_pred_proba_ht, axis=1)
        elif len(encoder_ht.classes_) == 1:
            y_pred_ht = [encoder_ht.classes_[0]] * n_items
            probabilities_ht = [1.0] * n_items
        else:
            errors = [f"Lỗi dự đoán HT: Model {model_name} không tồn tại"] * n_items

    except Exception as e:
        logger.error(f"Client {client_id}, Model {model_name}: Lỗi khi dự đoán HachToan batch: {e}", exc_info=True)
        errors = [f"Lỗi dự đoán HachToan model {model_name}: {e}"] * n_items

    for i in range(n_items):
        prob = probabilities_ht[i]
        if isinstance(prob, np.float64): prob = float(prob)
        results.append({
            config.TARGET_HACHTOAN: y_pred_ht[i],
            f"{config.TARGET_HACHTOAN}_prob": prob,
            "is_outlier_input1": outlier_flags_1[i],
            "error": errors[i]
        })
    return results


def _predict_mahanghoa_batch(
    client_id: str,
    model_name: str,  # THÊM PARAMETER MỚI
    input_df_mh: pd.DataFrame,
    preprocessor_mh: Optional[ColumnTransformer],
    model_mh: Optional[Any],
    encoder_mh: Optional[LabelEncoder],
    outlier_detector_2: Optional[Any]
) -> List[Dict[str, Any]]:
    """Dự đoán chỉ MaHangHoa cho một batch DataFrame với model_name support."""
    results = []
    n_items = len(input_df_mh)
    y_pred_mh = [None] * n_items
    probabilities_mh = [None] * n_items
    outlier_flags_2 = [False] * n_items
    errors = [None] * n_items

    if not preprocessor_mh or not encoder_mh:
        logger.error(f"Client {client_id}, Model {model_name}: Thiếu preprocessor hoặc encoder MaHangHoa.")
        for i in range(n_items):
            results.append({"error": f"Thiếu thành phần model MaHangHoa cho model {model_name}."})
        return results

    can_check_outlier_2 = bool(preprocessor_mh and outlier_detector_2)

    if config.TARGET_HACHTOAN not in input_df_mh.columns:
         logger.error(f"Client {client_id}, Model {model_name}: Input cho dự đoán MaHangHoa thiếu cột '{config.TARGET_HACHTOAN}'.")
         for i in range(n_items):
             results.append({"error": f"Thiếu input {config.TARGET_HACHTOAN} cho model {model_name}."})
         return results

    try:
        expected_features_mh = list(preprocessor_mh.feature_names_in_)
        input_data_aligned_mh = input_df_mh.reindex(columns=expected_features_mh, fill_value="")
        X_transformed_mh = preprocessor_mh.transform(input_data_aligned_mh)

        if can_check_outlier_2:
            outlier_flags_2 = check_outlier(outlier_detector_2, X_transformed_mh)

        if model_mh:
            y_pred_encoded_mh = model_mh.predict(X_transformed_mh)
            y_pred_proba_mh = model_mh.predict_proba(X_transformed_mh)
            y_pred_mh = encoder_mh.inverse_transform(y_pred_encoded_mh)
            probabilities_mh = np.max(y_pred_proba_mh, axis=1)
        elif len(encoder_mh.classes_) == 1:
            y_pred_mh = [encoder_mh.classes_[0]] * n_items
            probabilities_mh = [1.0] * n_items
        else:
            errors = [f"Lỗi dự đoán MH: Model {model_name} không tồn tại"] * n_items

    except Exception as e:
        logger.error(f"Client {client_id}, Model {model_name}: Lỗi khi dự đoán MaHangHoa batch: {e}", exc_info=True)
        errors = [f"Lỗi dự đoán MaHangHoa model {model_name}: {e}"] * n_items

    for i in range(n_items):
        prob = probabilities_mh[i]
        if isinstance(prob, np.float64): prob = float(prob)
        results.append({
            config.TARGET_MAHANGHOA: y_pred_mh[i],
            f"{config.TARGET_MAHANGHOA}_prob": prob,
            "is_outlier_input2": outlier_flags_2[i],
            "error": errors[i]
        })
    return results


def predict_combined(
        tax_code: str,  # PARAMETER 1: Mã số thuế
        model_type: str,  # PARAMETER 2: IN hoặc OUT
        input_data: pd.DataFrame  # PARAMETER 3: Dữ liệu input
) -> List[Dict[str, Any]]:
    """Thực hiện dự đoán kết hợp HachToan -> MaHangHoa."""
    logger.info(f"Bắt đầu dự đoán kết hợp cho tax code {tax_code}, model {model_type} với {len(input_data)} bản ghi.")

    # Validate model_type
    if model_type not in config.VALID_MODEL_NAMES:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be one of {config.VALID_MODEL_NAMES}")

    models_path = get_client_models_path(tax_code)

    # Load HachToan components với model_type
    preprocessor_ht_file = config.get_preprocessor_hachtoan_filename(model_type)
    hachtoan_model_file = config.get_hachtoan_model_filename(model_type)
    hachtoan_encoder_file = config.get_hachtoan_encoder_filename(model_type)
    outlier_detector_1_file = config.get_outlier_detector_1_filename(model_type)

    preprocessor_ht = load_joblib(models_path / preprocessor_ht_file)
    model_ht = load_joblib(models_path / hachtoan_model_file)
    encoder_ht = _load_label_encoder(tax_code, hachtoan_encoder_file)
    outlier_detector_1 = load_outlier_detector(tax_code, outlier_detector_1_file)

    # Load MaHangHoa components với model_type
    preprocessor_mh_file = config.get_preprocessor_mahanghoa_filename(model_type)
    mahanghoa_model_file = config.get_mahanghoa_model_filename(model_type)
    mahanghoa_encoder_file = config.get_mahanghoa_encoder_filename(model_type)
    outlier_detector_2_file = config.get_outlier_detector_2_filename(model_type)

    preprocessor_mh = load_joblib(models_path / preprocessor_mh_file)
    model_mh = load_joblib(models_path / mahanghoa_model_file)
    encoder_mh = _load_label_encoder(tax_code, mahanghoa_encoder_file)
    outlier_detector_2 = load_outlier_detector(tax_code, outlier_detector_2_file)

    # Predict HachToan first với updated signature
    results_ht = _predict_hachtoan_batch(
        tax_code, model_type, input_data, preprocessor_ht, model_ht, encoder_ht, outlier_detector_1
    )

    final_results = []
    indices_to_predict_mh = []
    input_list_mh = []

    for i, res_ht in enumerate(results_ht):
        hachtoan_pred = res_ht.get(config.TARGET_HACHTOAN)
        current_result = {
            config.TARGET_HACHTOAN: hachtoan_pred,
            f"{config.TARGET_HACHTOAN}_prob": res_ht.get(f"{config.TARGET_HACHTOAN}_prob"),
            config.TARGET_MAHANGHOA: None,
            f"{config.TARGET_MAHANGHOA}_prob": None,
            "is_outlier_input1": res_ht.get("is_outlier_input1", False),
            "is_outlier_input2": False,
            "error": res_ht.get("error")
        }

        # Check if should predict MaHangHoa
        if (hachtoan_pred is not None and
                isinstance(hachtoan_pred, str) and
                hachtoan_pred.startswith(tuple(config.HACHTOAN_PREFIX_FOR_MAHANGHOA)) and
                preprocessor_mh and encoder_mh):

            indices_to_predict_mh.append(i)
            try:
                input_dict = input_data.iloc[i].to_dict()
                input_dict[config.TARGET_HACHTOAN] = hachtoan_pred
                input_list_mh.append(input_dict)
            except IndexError:
                logger.error(f"IndexError khi truy cập input_data.iloc[{i}]")
                current_result["error"] = (current_result["error"] or "") + "; Lỗi lấy dữ liệu gốc cho MH"

        final_results.append(current_result)

    # Predict MaHangHoa if needed
    if indices_to_predict_mh and input_list_mh:
        logger.info(
            f"Tax code {tax_code}, Model {model_type}: Dự đoán MaHangHoa cho {len(indices_to_predict_mh)} bản ghi.")
        try:
            input_df_mh = pd.DataFrame(input_list_mh)
            results_mh = _predict_mahanghoa_batch(
                tax_code, model_type, input_df_mh, preprocessor_mh, model_mh, encoder_mh, outlier_detector_2
            )

            for idx, original_index in enumerate(indices_to_predict_mh):
                if idx < len(results_mh):
                    res_mh = results_mh[idx]
                    final_results[original_index][config.TARGET_MAHANGHOA] = res_mh.get(config.TARGET_MAHANGHOA)
                    final_results[original_index][f"{config.TARGET_MAHANGHOA}_prob"] = res_mh.get(
                        f"{config.TARGET_MAHANGHOA}_prob")
                    final_results[original_index]["is_outlier_input2"] = res_mh.get("is_outlier_input2", False)
                    if res_mh.get("error"):
                        if final_results[original_index]["error"]:
                            final_results[original_index]["error"] += f"; Lỗi MH: {res_mh['error']}"
                        else:
                            final_results[original_index]["error"] = f"Lỗi MH: {res_mh['error']}"
                else:
                    logger.error(
                        f"Index mismatch khi gộp kết quả MaHangHoa: idx={idx}, len(results_mh)={len(results_mh)}")
                    final_results[original_index]["error"] = (final_results[original_index][
                                                                  "error"] or "") + "; Lỗi nội bộ khi gộp kết quả MH"

        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi xử lý batch MaHangHoa: {e}", exc_info=True)
            for original_index in indices_to_predict_mh:
                final_results[original_index]["error"] = (final_results[original_index][
                                                              "error"] or "") + f"; Lỗi batch MH: {e}"

    logger.info(f"Dự đoán kết hợp hoàn tất cho tax code {tax_code}, model {model_type}.")
    return final_results


def predict_hachtoan_only(
        tax_code: str,  # PARAMETER 1: Mã số thuế
        model_type: str,  # PARAMETER 2: IN hoặc OUT
        input_data: pd.DataFrame  # PARAMETER 3: Dữ liệu input
) -> List[Dict[str, Any]]:
    """Tải model và gọi hàm dự đoán batch chỉ cho HachToan."""
    logger.info(f"Bắt đầu dự đoán CHỈ HachToan cho tax code {tax_code}, model {model_type}.")

    # Validate model_type
    if model_type not in config.VALID_MODEL_NAMES:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be one of {config.VALID_MODEL_NAMES}")

    models_path = get_client_models_path(tax_code)

    # Load với model_type pattern
    preprocessor_ht_file = config.get_preprocessor_hachtoan_filename(model_type)
    hachtoan_model_file = config.get_hachtoan_model_filename(model_type)
    hachtoan_encoder_file = config.get_hachtoan_encoder_filename(model_type)
    outlier_detector_1_file = config.get_outlier_detector_1_filename(model_type)

    preprocessor_ht = load_joblib(models_path / preprocessor_ht_file)
    model_ht = load_joblib(models_path / hachtoan_model_file)
    encoder_ht = _load_label_encoder(tax_code, hachtoan_encoder_file)
    outlier_detector_1 = load_outlier_detector(tax_code, outlier_detector_1_file)

    return _predict_hachtoan_batch(
        tax_code, model_type, input_data, preprocessor_ht, model_ht, encoder_ht, outlier_detector_1
    )


def predict_mahanghoa_only(
        tax_code: str,  # PARAMETER 1: Mã số thuế
        model_type: str,  # PARAMETER 2: IN hoặc OUT
        input_data_with_hachtoan: pd.DataFrame  # PARAMETER 3: Dữ liệu input (có HachToan)
) -> List[Dict[str, Any]]:
    """Tải model và gọi hàm dự đoán batch chỉ cho MaHangHoa."""
    logger.info(f"Bắt đầu dự đoán CHỈ MaHangHoa cho tax code {tax_code}, model {model_type}.")

    # Validate model_type
    if model_type not in config.VALID_MODEL_NAMES:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be one of {config.VALID_MODEL_NAMES}")

    models_path = get_client_models_path(tax_code)

    # Load với model_type pattern
    preprocessor_mh_file = config.get_preprocessor_mahanghoa_filename(model_type)
    mahanghoa_model_file = config.get_mahanghoa_model_filename(model_type)
    mahanghoa_encoder_file = config.get_mahanghoa_encoder_filename(model_type)
    outlier_detector_2_file = config.get_outlier_detector_2_filename(model_type)

    preprocessor_mh = load_joblib(models_path / preprocessor_mh_file)
    model_mh = load_joblib(models_path / mahanghoa_model_file)
    encoder_mh = _load_label_encoder(tax_code, mahanghoa_encoder_file)
    outlier_detector_2 = load_outlier_detector(tax_code, outlier_detector_2_file)

    return _predict_mahanghoa_batch(
        tax_code, model_type, input_data_with_hachtoan, preprocessor_mh, model_mh, encoder_mh, outlier_detector_2
    )