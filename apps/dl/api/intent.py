import time

import numpy as np

from fastapi import APIRouter, status
from sqlalchemy import delete, select
from sqlalchemy.orm import joinedload

from apps.dl.classifier import predict_intents
from apps.dl.models import DLIntentAppeal
from apps.dl.schemas.intent import (
    DLIntentAppealCreateRequest,
    DLIntentAppealResponse,
    DLIntentAppealStatsResponse,
)
from config import SessionDep
from middlewares.auth import user_context


router = APIRouter(prefix="/intent")


@router.post("/forward", status_code=status.HTTP_201_CREATED)
async def generate_intent(
    user_request: DLIntentAppealCreateRequest,
    session: SessionDep,
) -> DLIntentAppealResponse:
    user = user_context.get()
    start = time.time()

    intents = predict_intents(user_request.message)
    time_taken = time.time() - start

    request_obj = DLIntentAppeal(
        user_id=user.id,
        message=user_request.message,
        intents=intents,
        time_taken=time_taken,
    )

    session.add(request_obj)
    await session.commit()
    await session.refresh(request_obj)

    return {
        "id": request_obj.id,
        "user": user,
        "message": user_request.message,
        "intents": request_obj.intents,
        "time_taken": request_obj.time_taken,
        "created_at": request_obj.created_at,
        "updated_at": request_obj.updated_at,
    }


@router.get("/history", status_code=status.HTTP_200_OK)
async def get_history(session: SessionDep) -> list[DLIntentAppealResponse]:
    stmt = select(DLIntentAppeal).options(joinedload(DLIntentAppeal.user))
    result = await session.execute(stmt)

    return result.scalars().all()


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def erase_history(session: SessionDep):  # noqa: ANN201
    stmt = delete(DLIntentAppeal)
    await session.execute(stmt)
    await session.commit()


@router.get("/stats", status_code=status.HTTP_200_OK)
async def get_stats(session: SessionDep) -> DLIntentAppealStatsResponse:
    stmt = select(DLIntentAppeal.time_taken, DLIntentAppeal.message).select_from(DLIntentAppeal)
    result = (await session.execute(stmt)).mappings().all()
    if not result:
        return DLIntentAppealStatsResponse()

    time_array = np.array([row["time_taken"] for row in result])
    time_q_50, time_q_95, time_q_99 = np.percentile(time_array, [50, 95, 99])

    msg_array = np.array([len(row["message"]) for row in result])
    msg_q_50, msg_q_95, msg_q_99 = np.percentile(msg_array, [50, 95, 99])

    return {
        "time_taken": {
            "min": np.round(time_array.min(), 3),
            "mean": np.round(time_array.mean(), 3),
            "max": np.round(time_array.max(), 3),
            "q_50": np.round(time_q_50, 3),
            "q_95": np.round(time_q_95, 3),
            "q_99": np.round(time_q_99, 3),
        },
        "msg_len": {
            "min": np.round(msg_array.min(), 3),
            "mean": np.round(msg_array.mean(), 3),
            "max": np.round(msg_array.max(), 3),
            "q_50": np.round(msg_q_50, 3),
            "q_95": np.round(msg_q_95, 3),
            "q_99": np.round(msg_q_99, 3),
        },
    }
