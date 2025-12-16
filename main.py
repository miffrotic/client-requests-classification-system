import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.users.api import router as users_router
from config import settings


app = FastAPI(
    docs_url=f"{settings.app.URL_PREFIX}/docs/",
    redoc_url=f"{settings.app.URL_PREFIX}/redoc/",
    openapi_url=f"{settings.app.URL_PREFIX}/openapi.json",
    title="MyHotelki API",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"])

app.include_router(users_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
