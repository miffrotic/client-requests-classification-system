from fastapi import APIRouter
from config import settings
from apps.users.api.profiles import router as profiles_router
from apps.users.api.files import router as files_router


router = APIRouter(prefix=f"{settings.app.URL_PREFIX}/users", tags=["Users"])

router.include_router(profiles_router)
router.include_router(files_router)
