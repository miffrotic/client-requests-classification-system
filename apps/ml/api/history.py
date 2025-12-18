import time

import numpy as np

from fastapi import APIRouter, status
from sqlalchemy import delete, select
from sqlalchemy.orm import joinedload

from apps.ml.models import MLIntentAppeal
from apps.ml.schemas.history import (
    MLIntentAppealCreateRequest,
    MLIntentAppealResponse,
    MLIntentAppealStatsResponse,
)
from apps.ml.utils.processors import bow_vec, mlb, model, preprocessor
from config import SessionDep
from middlewares.auth import user_context


router = APIRouter(prefix="/intent")


@router.post("/forward", status_code=status.HTTP_201_CREATED)
async def generate_intent(
    user_request: MLIntentAppealCreateRequest,
    session: SessionDep,
) -> MLIntentAppealResponse:
    user = user_context.get()

    start = time.time()

    preporcessed_message = preprocessor.clean_text(user_request.message)
    vectorized_message = bow_vec.transform([preporcessed_message]).toarray()
    prediction = model.predict(vectorized_message)
    intents = ", ".join(mlb.classes_[prediction[0] == 1])

    time_taken = time.time() - start

    request_obj = MLIntentAppeal(
        user_id=user.id,
        message=user_request.message,
        intents=intents or "unknown intent",
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
async def get_history(session: SessionDep) -> list[MLIntentAppealResponse]:
    stmt = select(MLIntentAppeal).options(joinedload(MLIntentAppeal.user))

    result = await session.execute(stmt)

    return result.scalars().all()


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def erase_history(session: SessionDep):  # noqa: ANN201
    stmt = delete(MLIntentAppeal)

    await session.execute(stmt)
    await session.commit()


@router.get("/stats", status_code=status.HTTP_200_OK)
async def get_stats(session: SessionDep) -> MLIntentAppealStatsResponse:
    stmt = select(MLIntentAppeal.time_taken, MLIntentAppeal.message).select_from(MLIntentAppeal)

    result = (await session.execute(stmt)).mappings().all()
    if not result:
        return MLIntentAppealStatsResponse()

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
