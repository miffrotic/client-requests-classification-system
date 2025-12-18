from fastapi import APIRouter

from apps.ml.api.intent import router as intent_router
from config import settings


router = APIRouter(prefix=f"{settings.app.URL_PREFIX}/ml", tags=["ML"])

router.include_router(intent_router)
