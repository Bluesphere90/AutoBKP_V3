import logging
import re
import pandas as pd
from typing import List, Optional

# --- Xử lý tiếng Việt ---
# Chọn một thư viện: underthesea hoặc pyvi
# Đảm bảo đã cài đặt: pip install underthesea hoac pip install pyvi
try:
    from underthesea import word_tokenize
    VIETNAMESE_TOKENIZER = 'underthesea'
except ImportError:
    try:
        from pyvi import ViTokenizer
        VIETNAMESE_TOKENIZER = 'pyvi'
    except ImportError:
        VIETNAMESE_TOKENIZER = None
        logging.warning("Không tìm thấy thư viện 'underthesea' hoặc 'pyvi'. Xử lý tiếng Việt sẽ bị hạn chế.")

# --- Scikit-learn ---
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --- Project imports ---
from app.core.config import (
    INPUT_COLUMNS,
    TARGET_HACHTOAN,
    TARGET_MAHANGHOA,
    TFIDF_MAX_FEATURES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Bước tiền xử lý văn bản tiếng Việt ---

def preprocess_vietnamese_text(text_series: pd.Series) -> pd.Series:
    """Áp dụng các bước làm sạch và tách từ cho cột văn bản tiếng Việt."""
    # 1. Chuyển thành chữ thường
    processed_series = text_series.str.lower()
    # 2. Loại bỏ ký tự đặc biệt (giữ lại khoảng trắng và ký tự tiếng Việt)
    processed_series = processed_series.apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    # 3. (Tùy chọn) Loại bỏ số
    # processed_series = processed_series.apply(lambda x: re.sub(r'\d+', '', x))
    # 4. Tách từ
    if VIETNAMESE_TOKENIZER == 'underthesea':
        processed_series = processed_series.apply(lambda x: word_tokenize(x, format="text"))
    elif VIETNAMESE_TOKENIZER == 'pyvi':
        processed_series = processed_series.apply(lambda x: ViTokenizer.tokenize(x))
    else:
        # Fallback: tách từ cơ bản bằng khoảng trắng nếu không có thư viện
        logger.warning("Sử dụng tách từ cơ bản bằng khoảng trắng.")
        processed_series = processed_series.apply(lambda x: ' '.join(x.split()))

    # 5. (Tùy chọn) Loại bỏ stopwords - Cần danh sách stopwords tiếng Việt
    # ... (thêm logic loại bỏ stopwords nếu cần) ...

    return processed_series

# Tạo FunctionTransformer để tích hợp vào Pipeline
vietnamese_text_processor = FunctionTransformer(preprocess_vietnamese_text, validate=False)

# --- Định nghĩa các Pipeline tiền xử lý ---

def create_preprocessor(input_features: List[str], include_hachtoan_input: bool = False) -> ColumnTransformer:
    """
    Tạo ColumnTransformer để tiền xử lý các features.
    Linh hoạt với các cột input bổ sung.

    Args:
        input_features: Danh sách tên các cột input có trong DataFrame.
        include_hachtoan_input: True nếu HachToan là một input (cho model MaHangHoa).

    Returns:
        Một đối tượng ColumnTransformer đã cấu hình.
    """
    transformers = []

    # Xác định các cột cho từng loại xử lý
    text_features = ['TenHangHoaDichVu']
    categorical_features = ['TaxCode']
    if include_hachtoan_input and TARGET_HACHTOAN in input_features:
        categorical_features.append(TARGET_HACHTOAN)

    # Xác định các cột còn lại (có thể là số hoặc categorical khác)
    # Đây là phần cần linh hoạt để mở rộng
    processed_cols = set(text_features + categorical_features)
    remaining_cols = [col for col in input_features if col not in processed_cols]

    numerical_features = []
    other_categorical_features = []

    # Phân loại các cột còn lại (ví dụ đơn giản dựa trên tên)
    # Trong thực tế, có thể cần logic phức tạp hơn dựa trên dtype hoặc metadata
    for col in remaining_cols:
        # Ví dụ: nếu tên cột chứa 'SoLuong', 'Gia', 'ThanhTien' -> coi là số
        if any(keyword in col.lower() for keyword in ['soluong', 'gia', 'thanhtien', 'number', 'amount', 'value']):
            numerical_features.append(col)
        else:
            # Mặc định coi các cột còn lại là categorical (cần xem xét kỹ)
            other_categorical_features.append(col)

    # --- Định nghĩa các bước biến đổi ---

    # 1. Xử lý văn bản (TenHangHoaDichVu)
    if 'TenHangHoaDichVu' in input_features:
        text_pipeline = Pipeline([
            ('preprocess', vietnamese_text_processor),
            ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))) # xem xét thêm min_df, max_df
        ])
        transformers.append(('text_processing', text_pipeline, 'TenHangHoaDichVu'))
        logger.info(f"Đã thêm pipeline xử lý text cho cột: TenHangHoaDichVu")


    # 2. Xử lý Categorical đã biết (MSTNguoiBan, HachToan nếu có)
    if categorical_features:
        # Lọc ra các cột thực sự tồn tại trong input_features
        valid_categorical_features = [col for col in categorical_features if col in input_features]
        if valid_categorical_features:
            categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse=False có thể tốn bộ nhớ nếu nhiều feature
            transformers.append(('categorical_known', categorical_encoder, valid_categorical_features))
            logger.info(f"Đã thêm OneHotEncoder cho các cột: {valid_categorical_features}")


    # 3. Xử lý các cột Categorical khác (phát hiện được)
    if other_categorical_features:
        # Sử dụng OneHotEncoder cho các cột này, giả định số lượng không quá lớn
        # Cân nhắc dùng FeatureHasher nếu số lượng giá trị unique lớn
        other_categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers.append(('categorical_other', other_categorical_encoder, other_categorical_features))
        logger.info(f"Đã thêm OneHotEncoder cho các cột categorical khác: {other_categorical_features}")


    # 4. Xử lý các cột Số (phát hiện được)
    if numerical_features:
        numerical_transformer = StandardScaler() # Hoặc MinMaxScaler
        transformers.append(('numerical', numerical_transformer, numerical_features))
        logger.info(f"Đã thêm StandardScaler cho các cột số: {numerical_features}")


    # --- Tạo ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' # Bỏ qua các cột không được xử lý rõ ràng
                         # Hoặc 'passthrough' nếu muốn giữ lại (cẩn thận với kiểu dữ liệu)
    )

    logger.info(f"Đã tạo ColumnTransformer với {len(transformers)} bộ biến đổi.")
    return preprocessor

# --- Hàm tiện ích để tạo các preprocessor cụ thể ---

def create_hachtoan_preprocessor(all_input_features: List[str]) -> ColumnTransformer:
    """Tạo preprocessor cho mô hình dự đoán HachToan."""
    logger.info("Tạo preprocessor cho mô hình HachToan...")
    # Loại bỏ các cột target khỏi danh sách input cho preprocessor
    features_for_hachtoan = [
        f for f in all_input_features
        if f not in [TARGET_HACHTOAN, TARGET_MAHANGHOA]
    ]
    logger.info(f"Features sử dụng cho HachToan preprocessor: {features_for_hachtoan}")
    return create_preprocessor(features_for_hachtoan, include_hachtoan_input=False)

def create_mahanghoa_preprocessor(all_input_features: List[str]) -> ColumnTransformer:
    """Tạo preprocessor cho mô hình dự đoán MaHangHoa."""
    logger.info("Tạo preprocessor cho mô hình MaHangHoa...")
    # Bao gồm HachToan làm input, loại bỏ target MaHangHoa
    features_for_mahanghoa = [
        f for f in all_input_features
        if f != TARGET_MAHANGHOA # Giữ lại HachToan nếu có
    ]
    # Đảm bảo HachToan có trong list nếu nó tồn tại trong dataframe gốc
    if TARGET_HACHTOAN not in features_for_mahanghoa and TARGET_HACHTOAN in all_input_features:
         features_for_mahanghoa.append(TARGET_HACHTOAN)

    logger.info(f"Features sử dụng cho MaHangHoa preprocessor: {features_for_mahanghoa}")
    return create_preprocessor(features_for_mahanghoa, include_hachtoan_input=True)