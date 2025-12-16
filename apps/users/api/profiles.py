from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import NoResultFound

from apps.users.models import UserProfile
from apps.users.schemas.profiles import UserProfileResponse
from config import SessionDep


router = APIRouter(prefix="/profiles", tags=["Profiles"])


@router.get("/", status_code=status.HTTP_200_OK)
async def get_user_profile(user_id: int, session: SessionDep) -> UserProfileResponse:
    try:
        return await session.get_one(UserProfile, user_id)
    except NoResultFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="User is not found")
