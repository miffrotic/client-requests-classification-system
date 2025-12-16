from fastapi import APIRouter

router = APIRouter(prefix="/profiles", tags=["Users Profiles"])


@router.get("/")
async def get_user_profile():
    return {"message": "Zero users"}
