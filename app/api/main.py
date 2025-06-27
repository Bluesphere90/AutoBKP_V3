from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# --- Import individual routers to avoid circular imports ---
from app.api.endpoints.training import router as training_router
from app.api.endpoints.prediction import router as prediction_router  
from app.api.endpoints.models import router as models_router

from app.core.config import API_TITLE, API_DESCRIPTION, API_VERSION

# --- Khởi tạo FastAPI App ---
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# --- Gắn các Routers ---
app.include_router(training_router, prefix="/training", tags=["Training"])
app.include_router(prediction_router, prefix="/prediction", tags=["Prediction"])
app.include_router(models_router, prefix="/models", tags=["Models"])

# --- Endpoint gốc (Optional) ---
@app.get("/", include_in_schema=False)
async def root():
    """Chuyển hướng đến trang tài liệu API (Swagger UI)."""
    return RedirectResponse(url="/docs")