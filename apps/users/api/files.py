from fastapi import APIRouter


router = APIRouter(prefix="/files", tags=["Users Files"])


@router.get("/")
async def get_user_files() -> dict[str, str]:
    return {"message": "Zero files"}
