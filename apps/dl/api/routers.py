from fastapi import APIRouter

from apps.dl.api.intent import router as intent_router
from config import settings


router = APIRouter(prefix=f"{settings.app.URL_PREFIX}/dl", tags=["DL"])

router.include_router(intent_router)
