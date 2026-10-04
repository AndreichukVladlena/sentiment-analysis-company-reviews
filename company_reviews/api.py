"""FastAPI inference service. Run with uvicorn company_reviews.api:app."""

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from company_reviews.history import HistoryStore
from company_reviews.model_registry import (
    ModelLoadError,
    ModelRegistry,
    UnknownModelError,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_BODY_BYTES = 1_048_576


class ReviewRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Id: int | None = Field(
        default=None,
        ge=-(2**63),
        le=2**63 - 1,
        description="Необязательный целочисленный Id строки датасета; не используется моделью.",
    )
    Review: str = Field(
        min_length=1,
        max_length=20_000,
        description="Текст отзыва: 1–20 000 символов, содержит хотя бы один непробельный символ.",
        examples=["Fast delivery and helpful customer support."],
    )

    @field_validator("Review")
    @classmethod
    def nonblank_unicode_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Review не должен состоять только из пробелов")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Review должен содержать корректный Unicode") from None
        return value


ReviewBatch = Annotated[list[ReviewRow], Field(min_length=1, max_length=128)]


class Prediction(BaseModel):
    label: int = Field(
        ge=1, le=5, description="Оценка 1–5: медиана распределения модели."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Вероятность выбранной оценки; не гарантия правильности прогноза.",
    )


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model_id: str = Field(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description="Идентификатор из GET /models. Файлы и URL не принимаются.",
        examples=["tfidf"],
    )


class BodyLimitMiddleware:
    """Bound the body before JSON parsing, including chunked HTTP requests."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                response = JSONResponse(
                    status_code=413, content={"detail": "Тело запроса превышает 1 МиБ"}
                )
                await response(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


_VALIDATION_MESSAGES = {
    "missing": "Обязательное поле отсутствует",
    "string_type": "Ожидается строка",
    "int_type": "Ожидается целое число",
    "string_too_short": "Текст не должен быть пустым",
    "string_too_long": "Текст превышает 20 000 символов",
    "too_short": "Массив должен содержать хотя бы один отзыв",
    "too_long": "Массив должен содержать не более 128 отзывов",
    "extra_forbidden": "Неизвестное поле",
    "string_pattern_mismatch": "Недопустимый идентификатор модели; смотрите GET /models",
    "json_invalid": "Некорректный JSON",
    "greater_than_equal": "Число меньше допустимого значения",
    "less_than_equal": "Число больше допустимого значения",
    "model_attributes_type": "Ожидается объект с полем Review или массив таких объектов",
    "list_type": "Ожидается массив отзывов",
}


async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = []
    for error in exc.errors():
        location = list(error["loc"])
        # A union has two validation branches. Show only the submitted shape.
        if len(location) > 1:
            branch = str(location[1])
            if isinstance(exc.body, list) and branch == "ReviewRow":
                continue
            if isinstance(exc.body, dict) and branch.startswith("list["):
                continue
            if branch == "ReviewRow" or branch.startswith("list["):
                location.pop(1)
        message = _VALIDATION_MESSAGES.get(error["type"], "Некорректное значение")
        if error["type"] == "value_error":
            message = error["msg"].removeprefix("Value error, ")
        details.append({"loc": location, "message": message})
    return JSONResponse(status_code=422, content={"detail": details})


def create_app(
    model_dir: Path | None = None,
    history_db: Path | None = None,
    store_review_text: bool | None = None,
) -> FastAPI:
    """Create an app without loading model files until lifespan startup."""
    model_dir = Path(model_dir or os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
    history_db = Path(
        history_db or os.getenv("HISTORY_DB", PROJECT_ROOT / "data/history.sqlite3")
    )
    if store_review_text is None:
        store_review_text = os.getenv("STORE_REVIEW_TEXT", "0") == "1"

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.registry = ModelRegistry(model_dir)
        application.state.history = HistoryStore(history_db, store_review_text)
        yield

    application = FastAPI(
        title="Оценка отзывов компаний",
        version="1.0.0",
        description=(
            "Прогноз оценки отзыва от 1 до 5. Один объект возвращает один прогноз; "
            "массив — массив в исходном порядке. Модель и история загружаются при запуске. "
            "Сервис рассчитан на один процесс Uvicorn. Swagger: /docs."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(BodyLimitMiddleware)
    application.add_exception_handler(RequestValidationError, validation_error_handler)

    @application.get("/health", summary="Проверить готовность сервиса", tags=["Сервис"])
    def health():
        active = application.state.registry.snapshot()
        return {
            "status": "ok",
            "model_id": active.spec.id,
            "model_version": active.spec.version,
        }

    @application.get("/models", summary="Доступные локальные модели", tags=["Модели"])
    def models():
        registry = application.state.registry
        return {
            "active_model": registry.snapshot().spec.id,
            "models": [
                {
                    "id": spec.id,
                    "version": spec.version,
                    "description": spec.description,
                }
                for spec in registry.manifest.models
            ],
        }

    @application.post(
        "/load_model",
        summary="Переключить активную модель",
        tags=["Модели"],
        description=(
            "Выберите model_id из GET /models. Сервис проверит файл, версию sklearn "
            "и пробный прогноз, затем атомарно переключит модель. Ошибка загрузки "
            "сохраняет предыдущую модель. После перезапуска выбирается модель по умолчанию."
        ),
        responses={
            404: {"description": "Неизвестный model_id"},
            503: {"description": "Модель не прошла проверку"},
        },
    )
    def load_model(selection: ModelSelection):
        try:
            active = application.state.registry.load(selection.model_id)
        except UnknownModelError:
            raise HTTPException(
                404, "Неизвестная модель; выберите model_id из GET /models"
            ) from None
        except ModelLoadError:
            LOGGER.exception("Registered model loading failed")
            raise HTTPException(
                503, "Модель не прошла проверку; предыдущая модель остаётся активной"
            ) from None
        return {"model_id": active.spec.id, "model_version": active.spec.version}

    @application.post(
        "/predict",
        response_model=Prediction | list[Prediction],
        summary="Предсказать оценку отзыва",
        tags=["Прогноз"],
        description=(
            "Принимает строку датасета {Id, Review} или массив из 1–128 строк. Id необязателен. "
            "Review: до 20 000 символов; суммарно до 200 000 символов на запрос. "
            "label — медиана распределения для минимизации MAE; confidence — вероятность "
            "именно этой оценки, которая может отличаться от наиболее вероятной оценки. "
            "Это не гарантия правильности. История успешных прогнозов сохраняется в SQLite; "
            "исходный текст по умолчанию не сохраняется. Заголовки X-Model-Id, X-Model-Version "
            "и X-Request-Id позволяют связать ответ с моделью и историей."
        ),
        responses={
            413: {"description": "Тело запроса больше 1 МиБ"},
            422: {"description": "Некорректные поля или превышены ограничения"},
            503: {"description": "Ошибка модели или записи истории"},
        },
    )
    def predict(
        payload: Annotated[
            ReviewRow | ReviewBatch,
            Body(
                openapi_examples={
                    "single": {
                        "summary": "Один отзыв",
                        "value": {
                            "Id": 1,
                            "Review": "Excellent service and fast delivery!",
                        },
                    },
                    "batch": {
                        "summary": "Два отзыва",
                        "value": [
                            {"Review": "Great service."},
                            {"Id": 2, "Review": "My order never arrived."},
                        ],
                    },
                }
            ),
        ],
        response: Response,
    ):
        single = isinstance(payload, ReviewRow)
        rows = [payload] if single else payload
        if sum(len(row.Review) for row in rows) > 200_000:
            raise HTTPException(
                422, "Суммарная длина Review превышает 200 000 символов"
            )
        active = application.state.registry.snapshot()
        start = perf_counter()
        try:
            predictions = active.predict([row.Review for row in rows])
        except Exception:
            LOGGER.exception("Model inference failed")
            raise HTTPException(503, "Модель не смогла обработать запрос") from None
        inference_ms = (perf_counter() - start) * 1000
        try:
            request_id = application.state.history.record(
                [(row.Id, row.Review) for row in rows],
                predictions,
                active.spec,
                inference_ms,
            )
        except (sqlite3.Error, OSError):
            LOGGER.exception("Request history could not be saved")
            raise HTTPException(
                503, "История временно недоступна; повторите запрос позже"
            ) from None
        response.headers["X-Model-Id"] = active.spec.id
        response.headers["X-Model-Version"] = active.spec.version
        response.headers["X-Request-Id"] = request_id
        return predictions[0] if single else predictions

    return application


app = create_app()
