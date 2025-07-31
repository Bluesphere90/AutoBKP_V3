# app/core/config.py

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum # Thêm Enum

# Load environment variables from .env file if it exists
load_dotenv()
logger = logging.getLogger(__name__) # Thêm logger

# --- Paths ---
BASE_DATA_PATH = Path(os.getenv("DATA_PATH", "/app/data/client_data"))
BASE_MODELS_PATH = Path(os.getenv("MODELS_PATH", "/app/models/client_models"))
BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)
BASE_MODELS_PATH.mkdir(parents=True, exist_ok=True)

# --- Filenames ---
TRAINING_DATA_FILENAME = "training_data.csv"
INCREMENTAL_DATA_PREFIX = "incremental_data_"
METADATA_FILENAME_PREFIX = "metadata_" # Thêm prefix cho metadata
# Tên file cố định cho model/preprocessor/outlier/encoder
PREPROCESSOR_HACHTOAN_FILENAME = "preprocessor_hachtoan.joblib"
PREPROCESSOR_MAHANGHOA_FILENAME = "preprocessor_mahanghoa.joblib"
HACHTOAN_MODEL_FILENAME = "hachtoan_model.joblib"
MAHANGHOA_MODEL_FILENAME = "mahanghoa_model.joblib"
HACHTOAN_ENCODER_FILENAME = "hachtoan_encoder.joblib"
MAHANGHOA_ENCODER_FILENAME = "mahanghoa_encoder.joblib"
OUTLIER_DETECTOR_1_FILENAME = "outlier_detector_1.joblib"
OUTLIER_DETECTOR_2_FILENAME = "outlier_detector_2.joblib"
LABEL_ENCODERS_DIR = "label_encoders"

# --- Column Names ---
INPUT_COLUMNS = ["TaxCode", "TenHangHoaDichVu"]
TARGET_HACHTOAN = "HachToan"
TARGET_MAHANGHOA = "MaHangHoa"

# --- Model Logic ---
HACHTOAN_PREFIX_FOR_MAHANGHOA = ["15", "51"]

# --- Model Selection & Parameters ---
# Sử dụng Enum để định nghĩa các loại model hỗ trợ
class SupportedModels(str, Enum):
    RANDOM_FOREST = "RandomForestClassifier"
    LOGISTIC_REGRESSION = "LogisticRegression"
    MULTINOMIAL_NB = "MultinomialNB"
    LINEAR_SVC = "LinearSVC"
    # Thêm các model khác nếu muốn hỗ trợ, ví dụ:
    # LIGHTGBM = "LGBMClassifier"

# Tham số mặc định cho từng loại model
DEFAULT_MODEL_PARAMS = {
    SupportedModels.RANDOM_FOREST: {
        "n_estimators": 100, "random_state": 42, "class_weight": "balanced",
        "n_jobs": -1, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1
    },
    SupportedModels.LOGISTIC_REGRESSION: {
        "multi_class": 'ovr', "solver": 'liblinear', "random_state": 42,
        "class_weight": 'balanced', "max_iter": 1000, "C": 1.0
    },
    SupportedModels.MULTINOMIAL_NB: {
        "alpha": 1.0 # Tham số smoothing
    },
    SupportedModels.LINEAR_SVC: {
        "C": 1.0, "random_state": 42, "class_weight": "balanced", "max_iter": 2000,
        "dual": "auto" # Thêm dual='auto' để tránh warning
    }
    # Thêm params cho các model khác
    # SupportedModels.LIGHTGBM: {"n_estimators": 100, ...}
}

# Đọc cấu hình model từ biến môi trường (ưu tiên hơn giá trị mặc định)
# Biến này chỉ dùng làm mặc định nếu không có gì được chọn/lưu
DEFAULT_MODEL_TYPE = SupportedModels.RANDOM_FOREST # Chọn RF làm mặc định nếu không có gì khác
SELECTED_MODEL_NAME_FROM_ENV = os.getenv("SELECTED_MODEL_NAME", DEFAULT_MODEL_TYPE.value)

# --- Other ML Params ---
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", 5000))
OUTLIER_CONTAMINATION_STR = os.getenv("OUTLIER_CONTAMINATION", 'auto')
try:
    # Cố gắng chuyển đổi sang float nếu là số
    OUTLIER_CONTAMINATION = float(OUTLIER_CONTAMINATION_STR)
except ValueError:
    # Nếu không phải số (ví dụ: 'auto'), giữ nguyên dạng string
    OUTLIER_CONTAMINATION = OUTLIER_CONTAMINATION_STR

VALIDATION_SET_SIZE = float(os.getenv("VALIDATION_SET_SIZE", 0.2)) # Thêm cấu hình tỷ lệ validation

# --- API Configuration ---
API_TITLE = "HachToan & MaHangHoa Prediction API"
API_DESCRIPTION = "API for training and predicting HachToan and MaHangHoa using ML."
API_VERSION = "0.1.0"

# --- Model Names ---
VALID_MODEL_NAMES = ["IN", "OUT"]  # Chỉ cho phép 2 loại model

# --- Updated Filenames với model name support ---
def get_hachtoan_model_filename(model_name: str) -> str:
    """Tạo tên file cho HachToan model."""
    return f"hachtoan_model_{model_name}.joblib"

def get_mahanghoa_model_filename(model_name: str) -> str:
    """Tạo tên file cho MaHangHoa model."""
    return f"mahanghoa_model_{model_name}.joblib"

def get_preprocessor_hachtoan_filename(model_name: str) -> str:
    """Tạo tên file cho HachToan preprocessor."""
    return f"preprocessor_hachtoan_{model_name}.joblib"

def get_preprocessor_mahanghoa_filename(model_name: str) -> str:
    """Tạo tên file cho MaHangHoa preprocessor."""
    return f"preprocessor_mahanghoa_{model_name}.joblib"

def get_hachtoan_encoder_filename(model_name: str) -> str:
    """Tạo tên file cho HachToan encoder."""
    return f"hachtoan_encoder_{model_name}.joblib"

def get_mahanghoa_encoder_filename(model_name: str) -> str:
    """Tạo tên file cho MaHangHoa encoder."""
    return f"mahanghoa_encoder_{model_name}.joblib"

def get_outlier_detector_1_filename(model_name: str) -> str:
    """Tạo tên file cho outlier detector 1."""
    return f"outlier_detector_1_{model_name}.joblib"

def get_outlier_detector_2_filename(model_name: str) -> str:
    """Tạo tên file cho outlier detector 2."""
    return f"outlier_detector_2_{model_name}.joblib"

def get_metadata_filename(model_name: str, timestamp_str: str) -> str:
    """Tạo tên file cho metadata."""
    return f"metadata_{model_name}_{timestamp_str}.json"

HACHTOAN_FILTER_RULES = {
    "IN": {
        "description": "Exclude HachToan starting with '5' or '7'",
        "exclude_prefixes": ['5', '7']
    },
    "OUT": {
        "description": "Only HachToan starting with '5' or '7'",
        "include_prefixes": ['5', '7']
    }
}