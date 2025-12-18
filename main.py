import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.ml.api import router as ml_router
from apps.users.api import router as users_router
from config import settings
from middlewares.auth import auth_middleware


app = FastAPI(
    docs_url=settings.app.PUBLIC_URLS["docs_url"],
    redoc_url=settings.app.PUBLIC_URLS["redoc_url"],
    openapi_url=settings.app.PUBLIC_URLS["openapi_url"],
    title="Project API",
)

app.middleware("http")(auth_middleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

app.include_router(users_router)
app.include_router(ml_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
