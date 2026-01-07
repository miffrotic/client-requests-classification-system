from fastapi import APIRouter

from apps.users.api.auth import router as auth_router
from apps.users.api.profiles import router as profiles_router
from config import settings


router = APIRouter(prefix=f"{settings.app.URL_PREFIX}/users", tags=["Users"])

router.include_router(profiles_router)
router.include_router(auth_router)
