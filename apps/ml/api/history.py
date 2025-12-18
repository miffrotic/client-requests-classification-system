import pickle

from pathlib import Path

from fastapi import APIRouter, status

from apps.ml.models import RequestsHistory
from apps.ml.schemas.history import HistoryCreateRequest, HistoryCreateResponse
from apps.ml.utils.preprocessor import preprocessor
from config import BASE_DIR, SessionDep
from middlewares.auth import user_context


bow_vec_path = BASE_DIR / "pkl_models" / "bow_vectorizer_kgl.pkl"
model_path = BASE_DIR / "pkl_models" / "bow_linear_svc_model.pkl"
mlb_path = BASE_DIR / "pkl_models" / "multi_label_binarizer.pkl"

with Path(bow_vec_path).open("rb") as file:
    bow_vec = pickle.load(file)
with Path(model_path).open("rb") as file:
    model = pickle.load(file)
with Path(mlb_path).open("rb") as file:
    mlb = pickle.load(file)

router = APIRouter(prefix="/forward")


@router.post("/", status_code=status.HTTP_201_CREATED)
async def generate_intent(
    user_request: HistoryCreateRequest,
    session: SessionDep,
) -> HistoryCreateResponse:
    user = user_context.get()

    preporcessed_message = preprocessor.clean_text(user_request.message)
    vectorized_message = bow_vec.transform([preporcessed_message]).toarray()
    prediction = model.predict(vectorized_message)
    intents = ", ".join(mlb.classes_[prediction[0] == 1])

    request_obj = RequestsHistory(
        user_id=user.id,
        message=user_request.message,
        intents=intents or "unknown intent",
    )

    session.add(request_obj)
    await session.commit()
    await session.refresh(request_obj)

    return {
        "id": request_obj.id,
        "user": user,
        "message": user_request.message,
        "intents": request_obj.intents,
        "created_at": request_obj.created_at,
        "updated_at": request_obj.updated_at,
    }
