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

# --- Helper Functions (make_json_serializable, _fit/load_label_encoder - Giữ nguyên) ---
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
def train_client_models(client_id: str, initial_model_type_str: Optional[str] = None) -> bool:
    start_time = time.time()
    training_timestamp_utc = datetime.now(timezone.utc)
    logger.info(f"===== Bắt đầu quy trình huấn luyện cho client: {client_id} lúc {training_timestamp_utc} UTC =====")
    models_path = get_client_models_path(client_id)
    metadata = { # Khởi tạo metadata
        "client_id": client_id, "training_timestamp_utc": training_timestamp_utc.isoformat(),
        "status": "STARTED", "training_type": None, "selected_model_type": None,
        "data_info": {}, "hachtoan_model_info": {}, "mahanghoa_model_info": {},
        "training_duration_seconds": None, "error_message": None
    }

    # --- Xác định Training Type và Model Type ---
    latest_metadata_file = _find_latest_metadata_file(models_path)
    model_type_to_train = None
    if initial_model_type_str:
        metadata["training_type"] = "initial"
        if initial_model_type_str in [e.value for e in config.SupportedModels]:
            model_type_to_train = initial_model_type_str
            metadata["selected_model_type"] = model_type_to_train
            logger.info(f"Huấn luyện ban đầu, sử dụng model: {model_type_to_train}")
        else:
            error_msg = f"Loại model '{initial_model_type_str}' không hợp lệ.";
            logger.error(error_msg)
            metadata["status"] = "FAILED";
            metadata["error_message"] = error_msg;  # Lưu lỗi và thoát sớm
            # ... (lưu metadata lỗi) ...
            return False
    elif latest_metadata_file:
        metadata["training_type"] = "incremental"
        saved_model_type = _load_model_type_from_metadata(latest_metadata_file)
        if saved_model_type:
            model_type_to_train = saved_model_type
        else:
            model_type_to_train = config.DEFAULT_MODEL_TYPE.value; logger.warning(
                "Không đọc được model type, dùng default.")
        metadata["selected_model_type"] = model_type_to_train
        logger.info(f"Huấn luyện tăng cường, sử dụng model: {model_type_to_train}")
    else:
        error_msg = "Không thể xác định loại model.";
        logger.error(error_msg)
        metadata["status"] = "FAILED";
        metadata["error_message"] = error_msg;  # Lưu lỗi và thoát sớm
        # ... (lưu metadata lỗi) ...
        return False

    # --- Xác định data files (Giữ nguyên logic) ---
    client_data_path = config.BASE_DATA_PATH / client_id
    initial_data_file = client_data_path / config.TRAINING_DATA_FILENAME
    incremental_files = sorted(client_data_path.glob(f"{config.INCREMENTAL_DATA_PREFIX}*.csv"))
    data_files_used = []
    if initial_data_file.exists(): data_files_used.append(initial_data_file.name)
    if incremental_files: data_files_used.extend([f.name for f in incremental_files])
    if not data_files_used: metadata["training_type"] = "unknown"
    metadata["data_info"]["files_used"] = data_files_used


    # 1. Load tất cả dữ liệu (Giữ nguyên)
    df = load_all_client_data(client_id)
    if df is None or df.empty:
        error_msg = f"Không có dữ liệu huấn luyện hợp lệ.";
        logger.error(error_msg)
        metadata["status"] = "FAILED";
        metadata["error_message"] = error_msg;  # Lưu lỗi và thoát sớm
        # ... (lưu metadata lỗi) ...
        return False

    metadata["data_info"]["total_samples_loaded"] = len(df)
    metadata["data_info"]["columns_present"] = df.columns.tolist()

    # --- Khởi tạo các đối tượng model dựa trên model_type_to_train ---
    model_ht_instance = _instantiate_model(model_type_to_train)
    model_mh_instance = _instantiate_model(model_type_to_train) # Dùng cùng loại model cho cả hai

    if model_ht_instance is None: # Nếu không khởi tạo được model HachToan
        error_msg = f"Không thể khởi tạo model loại '{model_type_to_train}' cho HachToan."
        logger.error(error_msg); metadata["status"] = "FAILED"; metadata["error_message"] = error_msg
        # ... (Lưu metadata lỗi và return False) ...
        return False

    # Đặt tên file (Giữ nguyên)
    preprocessor_hachtoan_file = config.PREPROCESSOR_HACHTOAN_FILENAME
    hachtoan_model_save_file = config.HACHTOAN_MODEL_FILENAME # Tên file lưu cố định
    mahanghoa_model_save_file = config.MAHANGHOA_MODEL_FILENAME # Tên file lưu cố định
    hachtoan_encoder_file = config.HACHTOAN_ENCODER_FILENAME  # Cần tên này
    mahanghoa_encoder_file = config.MAHANGHOA_ENCODER_FILENAME  # Cần tên này
    outlier_detector_1_file = config.OUTLIER_DETECTOR_1_FILENAME  # Cần tên này
    outlier_detector_2_file = config.OUTLIER_DETECTOR_2_FILENAME  # Cần tên này
    preprocessor_mahanghoa_file = config.PREPROCESSOR_MAHANGHOA_FILENAME  # Cần tên này


    # 2. Huấn luyện mô hình HachToan (Truyền model_ht_instance)
    logger.info(f"--- Huấn luyện mô hình {model_type_to_train} cho HachToan ---")
    prep_ht, model_ht, enc_ht, metrics_ht = train_single_model(
        client_id=client_id, df=df, target_column=config.TARGET_HACHTOAN,
        preprocessor_creator=create_hachtoan_preprocessor,
        preprocessor_filename=preprocessor_hachtoan_file,
        model_object=model_ht_instance, # Truyền model đã khởi tạo
        model_save_filename=hachtoan_model_save_file, # Tên file để lưu
        encoder_filename=hachtoan_encoder_file,
        outlier_detector_filename=outlier_detector_1_file
    )

    # Lưu metadata HachToan (Giữ nguyên logic)
    metadata["hachtoan_model_info"]["preprocessor_saved"] = prep_ht is not None
    # ... (lưu các thông tin khác như trước) ...
    if model_ht: metadata["hachtoan_model_info"]["model_class"] = type(model_ht).__name__
    if model_ht: metadata["hachtoan_model_info"]["model_params"] = model_ht.get_params()
    metadata["hachtoan_model_info"]["evaluation_metrics"] = metrics_ht

    if prep_ht is None or enc_ht is None:
        error_msg = "Huấn luyện HT thất bại (prep/enc).";
        logger.error(error_msg)
        metadata["status"] = "FAILED";
        metadata["error_message"] = error_msg;  # Lưu lỗi và thoát sớm
        # ... (lưu metadata lỗi) ...
        return False

    # 3. Huấn luyện mô hình MaHangHoa (Truyền model_mh_instance)
    logger.info(f"--- Huấn luyện mô hình {model_type_to_train} cho MaHangHoa ---")
    metadata["mahanghoa_model_info"]["attempted"] = False
    df_mahanghoa = pd.DataFrame()  # Khởi tạo df rỗng

    # Chỉ thử lọc nếu cột MaHangHoa tồn tại trong df gốc
    if config.TARGET_MAHANGHOA in df.columns:
        logger.info(f"Lọc dữ liệu cho model MaHangHoa từ {len(df)} bản ghi...")
        # Lọc theo prefix HachToan
        df_filtered_prefix = df[
            df[config.TARGET_HACHTOAN].astype(str).str.startswith(tuple(config.HACHTOAN_PREFIX_FOR_MAHANGHOA))
        ].copy()

        if not df_filtered_prefix.empty:
            logger.info(
                f"Tìm thấy {len(df_filtered_prefix)} bản ghi có HachToan prefix '{config.HACHTOAN_PREFIX_FOR_MAHANGHOA}'.")
            # --- THỰC HIỆN dropna TRÊN MaHangHoa Ở ĐÂY ---
            original_len_mh = len(df_filtered_prefix)
            df_mahanghoa = df_filtered_prefix.dropna(subset=[config.TARGET_MAHANGHOA]).copy()
            # Cũng loại bỏ chuỗi rỗng '' nếu có trong MaHangHoa
            if not df_mahanghoa.empty:
                df_mahanghoa = df_mahanghoa[df_mahanghoa[config.TARGET_MAHANGHOA].astype(str) != '']

            dropped_count_mh = original_len_mh - len(df_mahanghoa)
            if dropped_count_mh > 0:
                logger.warning(
                    f"Đã loại bỏ {dropped_count_mh} hàng khi lọc cho MaHangHoa do thiếu giá trị target MaHangHoa.")
            # ------------------------------------------
        else:
            logger.info(f"Không tìm thấy bản ghi nào có HachToan prefix '{config.HACHTOAN_PREFIX_FOR_MAHANGHOA}'.")

    else:
        logger.warning(
            f"Cột '{config.TARGET_MAHANGHOA}' không tồn tại trong dữ liệu. Không thể lọc dữ liệu cho MaHangHoa.")

    metadata["data_info"]["samples_for_mahanghoa"] = len(df_mahanghoa)  # Ghi lại số mẫu sau khi lọc và dropna

    if not df_mahanghoa.empty:
        metadata["mahanghoa_model_info"]["attempted"] = True
        logger.info(f"Bắt đầu huấn luyện MaHangHoa với {len(df_mahanghoa)} bản ghi.")
        if config.TARGET_HACHTOAN not in df_mahanghoa.columns:
            logger.error(f"Dữ liệu lọc MaHangHoa thiếu cột {config.TARGET_HACHTOAN}.")
            metadata["mahanghoa_model_info"]["error"] = "Missing HachToan column."
        elif model_mh_instance is None:
            logger.error(f"Không thể khởi tạo model '{model_type_to_train}' cho MaHangHoa.")
            metadata["mahanghoa_model_info"]["error"] = f"Could not instantiate model."
        else:
            prep_mh, model_mh, enc_mh, metrics_mh = train_single_model(
                client_id=client_id, df=df_mahanghoa, target_column=config.TARGET_MAHANGHOA,
                preprocessor_creator=create_mahanghoa_preprocessor,
                preprocessor_filename=preprocessor_mahanghoa_file,
                model_object=model_mh_instance,
                model_save_filename=mahanghoa_model_save_file,
                encoder_filename=mahanghoa_encoder_file,
                outlier_detector_filename=outlier_detector_2_file
            )
            # Lưu metadata MaHangHoa (Giữ nguyên logic)
            metadata["mahanghoa_model_info"]["preprocessor_saved"] = prep_mh is not None
            # ... (lưu các thông tin khác) ...
            metadata["mahanghoa_model_info"]["evaluation_metrics"] = metrics_mh
            # ... (xử lý warning nếu model_mh is None) ...
    else:
        # Trường hợp không có dữ liệu để huấn luyện MH (do thiếu cột hoặc sau khi lọc/dropna)
        logger.warning(f"Không có dữ liệu hợp lệ để huấn luyện MaHangHoa.")
        if config.TARGET_MAHANGHOA in df.columns:  # Chỉ đánh dấu attempted nếu cột MH tồn tại ban đầu
            metadata["mahanghoa_model_info"]["attempted"] = True
        metadata["mahanghoa_model_info"]["message"] = "No suitable data found for training MaHangHoa."
        # Xóa file cũ của MaHangHoa (Giữ nguyên logic)
        if (models_path / mahanghoa_model_save_file).exists(): (models_path / mahanghoa_model_save_file).unlink()
        if (models_path / preprocessor_mahanghoa_file).exists(): (models_path / preprocessor_mahanghoa_file).unlink()
        # ... (xóa encoder, outlier) ...


    # --- Hoàn tất và Lưu Metadata ---
    end_time = time.time()
    metadata["training_duration_seconds"] = round(end_time - start_time, 2)
    metadata["status"] = "COMPLETED"

    meta_filename = f"{config.METADATA_FILENAME_PREFIX}{training_timestamp_utc.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        serializable_metadata = make_json_serializable(metadata)
        with open(models_path / meta_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_metadata, f, ensure_ascii=False, indent=4)
        logger.info(f"Đã lưu metadata huấn luyện vào {meta_filename}")
    except Exception as json_e:
        logger.error(f"Lỗi khi lưu metadata huấn luyện: {json_e}")

    logger.info(f"===== Quy trình huấn luyện cho client {client_id} đã hoàn tất ({metadata['training_duration_seconds']:.2f}s). =====")
    return (models_path / preprocessor_hachtoan_file).exists() and \
           (get_client_label_encoder_path(client_id) / hachtoan_encoder_file).exists()


# --- Prediction Logic (Refactored) ---
def _predict_hachtoan_batch(
    client_id: str,
    input_df: pd.DataFrame,
    preprocessor_ht: Optional[ColumnTransformer],
    model_ht: Optional[Any],
    encoder_ht: Optional[LabelEncoder],
    outlier_detector_1: Optional[Any]
) -> List[Dict[str, Any]]:
    """Dự đoán chỉ HachToan cho một batch DataFrame."""
    results = []
    n_items = len(input_df)
    y_pred_ht = [None] * n_items
    probabilities_ht = [None] * n_items
    outlier_flags_1 = [False] * n_items
    errors = [None] * n_items

    if not preprocessor_ht or not encoder_ht:
        logger.error(f"Client {client_id}: Thiếu preprocessor hoặc encoder HachToan.")
        for i in range(n_items): results.append({"error": "Thiếu thành phần model HachToan."})
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
            errors = ["Lỗi dự đoán HT: Model không tồn tại"] * n_items

    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi khi dự đoán HachToan batch: {e}", exc_info=True)
        errors = [f"Lỗi dự đoán HachToan: {e}"] * n_items

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
    input_df_mh: pd.DataFrame,
    preprocessor_mh: Optional[ColumnTransformer],
    model_mh: Optional[Any],
    encoder_mh: Optional[LabelEncoder],
    outlier_detector_2: Optional[Any]
) -> List[Dict[str, Any]]:
    """Dự đoán chỉ MaHangHoa cho một batch DataFrame (đã có HachToan input)."""
    results = []
    n_items = len(input_df_mh)
    y_pred_mh = [None] * n_items
    probabilities_mh = [None] * n_items
    outlier_flags_2 = [False] * n_items
    errors = [None] * n_items

    if not preprocessor_mh or not encoder_mh:
        logger.error(f"Client {client_id}: Thiếu preprocessor hoặc encoder MaHangHoa.")
        for i in range(n_items): results.append({"error": "Thiếu thành phần model MaHangHoa."})
        return results

    can_check_outlier_2 = bool(preprocessor_mh and outlier_detector_2)

    if config.TARGET_HACHTOAN not in input_df_mh.columns:
         logger.error(f"Client {client_id}: Input cho dự đoán MaHangHoa thiếu cột '{config.TARGET_HACHTOAN}'.")
         for i in range(n_items): results.append({"error": f"Thiếu input {config.TARGET_HACHTOAN}."})
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
            errors = ["Lỗi dự đoán MH: Model không tồn tại"] * n_items

    except Exception as e:
        logger.error(f"Client {client_id}: Lỗi khi dự đoán MaHangHoa batch: {e}", exc_info=True)
        errors = [f"Lỗi dự đoán MaHangHoa: {e}"] * n_items

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


def predict_combined(client_id: str, input_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Thực hiện dự đoán kết hợp HachToan -> MaHangHoa."""
    logger.info(f"Bắt đầu dự đoán kết hợp cho client {client_id} với {len(input_data)} bản ghi.")
    models_path = get_client_models_path(client_id)

    preprocessor_ht = load_joblib(models_path / "preprocessor_hachtoan.joblib")
    model_ht = load_joblib(models_path / config.HACHTOAN_MODEL_FILENAME)
    encoder_ht = _load_label_encoder(client_id, config.HACHTOAN_ENCODER_FILENAME)
    outlier_detector_1 = load_outlier_detector(client_id, config.OUTLIER_DETECTOR_1_FILENAME)

    preprocessor_mh = load_joblib(models_path / "preprocessor_mahanghoa.joblib")
    model_mh = load_joblib(models_path / config.MAHANGHOA_MODEL_FILENAME)
    encoder_mh = _load_label_encoder(client_id, config.MAHANGHOA_ENCODER_FILENAME)
    outlier_detector_2 = load_outlier_detector(client_id, config.OUTLIER_DETECTOR_2_FILENAME)

    results_ht = _predict_hachtoan_batch(
        client_id, input_data, preprocessor_ht, model_ht, encoder_ht, outlier_detector_1
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

        if (hachtoan_pred is not None and
            isinstance(hachtoan_pred, str) and
            hachtoan_pred.startswith(tuple(config.HACHTOAN_PREFIX_FOR_MAHANGHOA)) and
            preprocessor_mh and encoder_mh): # Cần prep và enc MH để xử lý MH

            indices_to_predict_mh.append(i)
            try:
                # Cẩn thận khi dùng iloc và to_dict, đảm bảo cột tồn tại
                input_dict = input_data.iloc[i].to_dict()
                input_dict[config.TARGET_HACHTOAN] = hachtoan_pred
                input_list_mh.append(input_dict)
            except IndexError:
                 logger.error(f"IndexError khi truy cập input_data.iloc[{i}]")
                 # Ghi lỗi vào kết quả hiện tại
                 current_result["error"] = (current_result["error"] or "") + "; Lỗi lấy dữ liệu gốc cho MH"


        final_results.append(current_result)


    if indices_to_predict_mh and input_list_mh: # Kiểm tra input_list_mh không rỗng
        logger.info(f"Client {client_id}: Dự đoán MaHangHoa cho {len(indices_to_predict_mh)} bản ghi.")
        try:
            input_df_mh = pd.DataFrame(input_list_mh)
            results_mh = _predict_mahanghoa_batch(
                client_id, input_df_mh, preprocessor_mh, model_mh, encoder_mh, outlier_detector_2
            )

            for idx, original_index in enumerate(indices_to_predict_mh):
                # Đảm bảo idx hợp lệ cho results_mh
                if idx < len(results_mh):
                    res_mh = results_mh[idx]
                    final_results[original_index][config.TARGET_MAHANGHOA] = res_mh.get(config.TARGET_MAHANGHOA)
                    final_results[original_index][f"{config.TARGET_MAHANGHOA}_prob"] = res_mh.get(f"{config.TARGET_MAHANGHOA}_prob")
                    final_results[original_index]["is_outlier_input2"] = res_mh.get("is_outlier_input2", False)
                    if res_mh.get("error"):
                        if final_results[original_index]["error"]:
                            final_results[original_index]["error"] += f"; Lỗi MH: {res_mh['error']}"
                        else:
                            final_results[original_index]["error"] = f"Lỗi MH: {res_mh['error']}"
                else:
                    logger.error(f"Index mismatch khi gộp kết quả MaHangHoa: idx={idx}, len(results_mh)={len(results_mh)}")
                    final_results[original_index]["error"] = (final_results[original_index]["error"] or "") + "; Lỗi nội bộ khi gộp kết quả MH"

        except Exception as e:
             logger.error(f"Lỗi nghiêm trọng khi xử lý batch MaHangHoa: {e}", exc_info=True)
             # Ghi lỗi vào tất cả các dòng lẽ ra phải dự đoán MH
             for original_index in indices_to_predict_mh:
                 final_results[original_index]["error"] = (final_results[original_index]["error"] or "") + f"; Lỗi batch MH: {e}"



    logger.info(f"Dự đoán kết hợp hoàn tất cho client {client_id}.")
    return final_results


def predict_hachtoan_only(client_id: str, input_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Tải model và gọi hàm dự đoán batch chỉ cho HachToan."""
    logger.info(f"Bắt đầu dự đoán CHỈ HachToan cho client {client_id}.")
    models_path = get_client_models_path(client_id)
    preprocessor_ht = load_joblib(models_path / "preprocessor_hachtoan.joblib")
    model_ht = load_joblib(models_path / config.HACHTOAN_MODEL_FILENAME)
    encoder_ht = _load_label_encoder(client_id, config.HACHTOAN_ENCODER_FILENAME)
    outlier_detector_1 = load_outlier_detector(client_id, config.OUTLIER_DETECTOR_1_FILENAME)

    return _predict_hachtoan_batch(
        client_id, input_data, preprocessor_ht, model_ht, encoder_ht, outlier_detector_1
    )


def predict_mahanghoa_only(client_id: str, input_data_with_hachtoan: pd.DataFrame) -> List[Dict[str, Any]]:
    """Tải model và gọi hàm dự đoán batch chỉ cho MaHangHoa."""
    logger.info(f"Bắt đầu dự đoán CHỈ MaHangHoa cho client {client_id}.")
    models_path = get_client_models_path(client_id)
    preprocessor_mh = load_joblib(models_path / "preprocessor_mahanghoa.joblib")
    model_mh = load_joblib(models_path / config.MAHANGHOA_MODEL_FILENAME)
    encoder_mh = _load_label_encoder(client_id, config.MAHANGHOA_ENCODER_FILENAME)
    outlier_detector_2 = load_outlier_detector(client_id, config.OUTLIER_DETECTOR_2_FILENAME)

    return _predict_mahanghoa_batch(
        client_id, input_data_with_hachtoan, preprocessor_mh, model_mh, encoder_mh, outlier_detector_2
    )