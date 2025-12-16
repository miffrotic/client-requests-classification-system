from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import NoResultFound

from apps.users.models import UserProfile
from apps.users.schemas.profiles import UserProfileCreateRequest, UserProfileResponse
from config import SessionDep


router = APIRouter(prefix="/profiles", tags=["Users Profiles"])


@router.post("/")
async def create_user_profile(
    user: UserProfileCreateRequest,
    session: SessionDep,
) -> UserProfileResponse:
    user = UserProfile(**user.model_dump())
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.get("/")
async def get_user_profile(user_id: int, session: SessionDep) -> UserProfileResponse:
    try:
        return await session.get_one(UserProfile, user_id)
    except NoResultFound:
        raise HTTPException(404, detail="Пользователь не найден")
