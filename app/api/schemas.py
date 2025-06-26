# app/api/schemas.py

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union

# --- Schemas Input gốc ---
class PredictionInputItem(BaseModel):
    """Schema cho một dòng dữ liệu input cơ bản."""
    MSTNguoiBan: str = Field(..., example="MST001", description="Mã số thuế người bán")
    TenHangHoaDichVu: str = Field(..., example="Phí vận chuyển tháng 12", description="Tên hàng hóa/dịch vụ")
    additional_data: Dict[str, Any] = Field({}, description="Dữ liệu bổ sung (key-value pairs)")

    def to_flat_dict(self) -> Dict[str, Any]:
        # Sử dụng model_dump trong Pydantic v2+
        data = self.model_dump(exclude={'additional_data'}, mode='python')
        data.update(self.additional_data)
        return data

class PredictionRequest(BaseModel):
    """Schema cho request dự đoán HachToan hoặc cả hai (endpoint gốc)."""
    items: List[PredictionInputItem] = Field(..., min_length=1, description="Danh sách các bản ghi cần dự đoán")

# --- Schema Input MỚI cho dự đoán chỉ MaHangHoa ---
class MaHangHoaPredictionInputItem(PredictionInputItem):
    """Schema input item khi chỉ dự đoán MaHangHoa (kế thừa và thêm HachToan)."""
    HachToan: str = Field(..., example="156", description="Hạch toán đã biết (dùng làm input)")

    # Kế thừa to_flat_dict từ PredictionInputItem là đủ

class MaHangHoaPredictionRequest(BaseModel):
    """Schema cho request chỉ dự đoán MaHangHoa."""
    items: List[MaHangHoaPredictionInputItem] = Field(..., min_length=1, description="Danh sách các bản ghi cần dự đoán MaHangHoa (phải có HachToan)")


# --- Schemas Output gốc ---
class PredictionResultItem(BaseModel):
    """Schema cho kết quả dự đoán đầy đủ (cả HachToan và MaHangHoa)."""
    HachToan: Optional[str] = Field(None, example="642", description="Kết quả dự đoán HachToan")
    HachToan_prob: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.85, description="Xác suất của HachToan dự đoán")
    MaHangHoa: Optional[str] = Field(None, example="VC001", description="Kết quả dự đoán MaHangHoa (nếu có)")
    MaHangHoa_prob: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.75, description="Xác suất của MaHangHoa dự đoán (nếu có)")
    is_outlier_input1: bool = Field(..., example=False, description="True nếu input cho model HachToan bị coi là outlier")
    is_outlier_input2: bool = Field(..., example=False, description="True nếu input cho model MaHangHoa bị coi là outlier")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có cho dòng này")

class PredictionResponse(BaseModel):
    """Schema cho response dự đoán đầy đủ."""
    results: List[PredictionResultItem] = Field(..., description="Danh sách kết quả dự đoán tương ứng với input")


# --- Schema Output MỚI chỉ cho HachToan ---
class HachToanPredictionResultItem(BaseModel):
    """Schema kết quả khi chỉ dự đoán HachToan."""
    HachToan: Optional[str] = Field(None, example="642", description="Kết quả dự đoán HachToan")
    HachToan_prob: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.85, description="Xác suất của HachToan dự đoán")
    is_outlier_input1: bool = Field(..., example=False, description="True nếu input cho model HachToan bị coi là outlier")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có cho dòng này")

class HachToanPredictionResponse(BaseModel):
    """Schema cho response chỉ dự đoán HachToan."""
    results: List[HachToanPredictionResultItem] = Field(..., description="Danh sách kết quả dự đoán HachToan")

# --- Schema Output MỚI chỉ cho MaHangHoa ---
class MaHangHoaPredictionResultItem(BaseModel):
    """Schema kết quả khi chỉ dự đoán MaHangHoa."""
    MaHangHoa: Optional[str] = Field(None, example="DELL01", description="Kết quả dự đoán MaHangHoa")
    MaHangHoa_prob: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.75, description="Xác suất của MaHangHoa dự đoán")
    is_outlier_input2: bool = Field(..., example=False, description="True nếu input cho model MaHangHoa bị coi là outlier")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có cho dòng này")

class MaHangHoaPredictionResponse(BaseModel):
    """Schema cho response chỉ dự đoán MaHangHoa."""
    results: List[MaHangHoaPredictionResultItem] = Field(..., description="Danh sách kết quả dự đoán MaHangHoa")


# --- Schemas cho Thông báo Chung ---
class MessageResponse(BaseModel):
    """Schema cho các response thông báo đơn giản (ví dụ: thành công/lỗi training)."""
    message: str = Field(..., example="Huấn luyện thành công cho client 'client_xyz'.")
    client_id: Optional[str] = None
    status_code: Optional[int] = 200

class ErrorDetail(BaseModel):
    """Schema chi tiết cho lỗi."""
    loc: Optional[List[Union[str, int]]] = None
    msg: str
    type: Optional[str] = None

class ErrorResponse(BaseModel):
    """Schema cho response lỗi chi tiết."""
    detail: Union[str, List[ErrorDetail]] = Field(..., description="Chi tiết lỗi")

# --- Schema cho Training Status ---
class TrainingStatusResponse(BaseModel):
    """Schema cho response kiểm tra tình trạng training đơn giản."""
    client_id: str = Field(..., example="client_abc")
    status: str = Field(..., example="COMPLETED", description="COMPLETED, FAILED, TRAINING, NOT_FOUND")
    training_type: Optional[str] = Field(None, example="initial", description="initial hoặc incremental")
    selected_model_type: Optional[str] = Field(None, example="RandomForestClassifier")
    training_timestamp: Optional[str] = Field(None, example="2025-01-15T10:30:00+00:00")
    duration_seconds: Optional[float] = Field(None, example=125.5)
    error_message: Optional[str] = Field(None, description="Thông báo lỗi nếu training failed")