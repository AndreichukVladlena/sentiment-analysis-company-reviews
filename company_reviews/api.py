"""FastAPI inference service. Run with uvicorn company_reviews.api:app."""

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from company_reviews.history import HistoryStore
from company_reviews.model_registry import (
    ModelLoadError,
    ModelRegistry,
    UnknownModelError,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ReviewRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Id: int | None = Field(
        default=None,
        ge=-(2**63),
        le=2**63 - 1,
        description="Необязательный целочисленный Id строки датасета, он не используется моделью.",
        examples=[1],
    )
    Review: str = Field(
        min_length=1,
        max_length=20_000,
        description=(
            "Англоязычный отзыв: 1–20 000 символов. "
            "Строка не должна состоять только из пробелов."
        ),
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
        ge=1,
        le=5,
        description="Оценка 1–5, выбранная как медиана распределения модели.",
        examples=[5],
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Оценённая моделью вероятность выбранной оценки.",
        examples=[0.98],
    )


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model_id: str = Field(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description="Идентификатор модели из GET /models, например tfidf или deberta.",
        examples=["tfidf"],
    )


class HistoryPrediction(Prediction):
    item_index: int = Field(
        ge=0, description="Позиция отзыва в запросе, начиная с 0.", examples=[0]
    )
    dataset_id: int | None = Field(
        description="Переданный Id отзыва; null, если Id не был указан.", examples=[1, None]
    )
    review_text: str | None = Field(
        description="Сохранённый текст отзыва; null, если сохранение было выключено.",
        examples=["Excellent service!", None],
    )


class HistoryRequest(BaseModel):
    request_id: str = Field(
        description="Идентификатор запроса из заголовка X-Request-Id.",
        examples=["a5c762ff96b945b7a4d619e530f026e8"],
    )
    created_at: str = Field(
        description="Время запроса в UTC; +00:00 обозначает часовой пояс UTC.",
        examples=["2026-10-08T19:59:55.079781+00:00"],
    )
    model_id: str = Field(description="Модель, выполнившая прогноз.", examples=["tfidf"])
    model_version: str = Field(
        description="Версия использованной модели.",
        examples=["2026-10-07T08:28:11+00:00"],
    )
    item_count: int = Field(description="Количество отзывов в запросе.", examples=[1])
    inference_ms: float = Field(
        description="Время прогнозирования всего запроса, мс; без записи в SQLite.",
        examples=[12.5],
    )
    predictions: list[HistoryPrediction] = Field(
        description="Прогнозы в порядке отзывов исходного запроса."
    )


class ModelState(BaseModel):
    model_id: str = Field(description="Активная модель.", examples=["tfidf"])
    model_version: str = Field(
        description="Версия загруженных файлов модели.",
        examples=["2026-10-07T08:28:11+00:00"],
    )


class HealthStatus(ModelState):
    status: Literal["ok"] = Field(
        description="API отвечает, активная модель загружена."
    )


class ModelInfo(BaseModel):
    id: str = Field(
        description="Идентификатор для POST /load_model.", examples=["tfidf"]
    )
    version: str = Field(
        description="Версия файлов модели.", examples=["2026-10-07T08:28:11+00:00"]
    )
    description: str = Field(
        description="Краткое описание модели.",
        examples=[
            "TF-IDF по словам и символам + 13 числовых признаков + Logistic Regression"
        ],
    )


class ModelCatalog(BaseModel):
    active_model: str = Field(description="Активная модель.", examples=["tfidf"])
    models: list[ModelInfo] = Field(
        description="Модели, зарегистрированные в каталоге.",
        examples=[
            [
                {
                    "id": "tfidf",
                    "version": "2026-10-07T08:28:11+00:00",
                    "description": "TF-IDF по словам и символам + 13 числовых признаков + Logistic Regression",
                },
                {
                    "id": "word-only",
                    "version": "2026-10-07T08:28:11+00:00",
                    "description": "Word and bigram TF-IDF",
                },
                {
                    "id": "deberta",
                    "version": "full-28e21afd362a",
                    "description": "DeBERTa-v3-base: финальное обучение на всех данных, 128 токенов",
                },
            ]
        ],
    )


class ErrorResponse(BaseModel):
    detail: str = Field(description="Причина ошибки.")


class ValidationIssue(BaseModel):
    loc: list[str | int] = Field(
        description="Путь к ошибке: тело запроса, поле или индекс элемента массива."
    )
    message: str = Field(description="Какое требование к данным нарушено.")


class ValidationErrorResponse(BaseModel):
    detail: list[ValidationIssue] = Field(description="Ошибки в данных запроса.")


async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Return useful field errors without echoing the submitted review text.
    details = [{"loc": error["loc"], "message": error["msg"]} for error in exc.errors()]
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
            "Прогноз оценки компании **от 1 до 5** по англоязычному отзыву.\n\n"
            "После запуска активна `tfidf`. Доступные модели перечислены в `GET /models`, "
            "для переключения используйте `POST /load_model`. "
            "История успешных прогнозов доступна в `GET /history`."
        ),
        openapi_tags=[
            {
                "name": "Прогноз",
                "description": "Оценка одного отзыва или массива отзывов.",
            },
            {
                "name": "Модели",
                "description": "Каталог моделей и выбор активной модели.",
            },
            {"name": "Сервис", "description": "Проверка готовности API."},
            {"name": "История", "description": "Сохранённые запросы и прогнозы."},
        ],
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 0,
            "displayRequestDuration": True,
        },
        lifespan=lifespan,
    )
    application.add_exception_handler(RequestValidationError, validation_error_handler)

    @application.get(
        "/health",
        response_model=HealthStatus,
        summary="Проверить готовность сервиса",
        description=(
            "Возвращает `ok`, идентификатор и версию активной модели. "
            "Запрос не выполняет прогноз и не записывается в историю."
        ),
        response_description="Сервис отвечает, активная модель загружена.",
        tags=["Сервис"],
    )
    def health():
        active = application.state.registry.snapshot()
        return {
            "status": "ok",
            "model_id": active.spec.id,
            "model_version": active.spec.version,
        }

    @application.get(
        "/models",
        response_model=ModelCatalog,
        summary="Посмотреть доступные модели",
        description=(
            "Возвращает каталог моделей и идентификатор активной модели. "
            "Передайте `id` выбранной модели в `POST /load_model`.\n\n"
            "Наличие модели в каталоге не подтверждает готовность её файлов: "
            "для `deberta` нужны скачанные веса и сборка с `MODEL_EXTRA=deberta`."
        ),
        response_description="Каталог моделей и текущий выбор.",
        tags=["Модели"],
    )
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

    @application.get(
        "/history",
        response_model=list[HistoryRequest],
        summary="Посмотреть историю прогнозов",
        tags=["История"],
        description=(
            "Возвращает историю успешных запросов к `POST /predict`, начиная с новых. "
            "Обращения к другим эндпоинтам и запросы с ошибками не включаются.\n\n"
            "**Просмотр порциями**\n\n"
            "- `limit`: сколько запросов вернуть, от 1 до 100; по умолчанию 10.\n"
            "- `offset`: сколько новых запросов пропустить; по умолчанию 0.\n"
            "- При `limit=10`: `offset=0` — первые 10, `offset=10` — следующие 10.\n"
            "- Пустая история или `offset` за её пределами возвращают `200` и `[]`.\n\n"
            "**Результат**\n\n"
            "Каждый элемент — один запрос с временем, моделью и массивом `predictions`. "
            "Прогнозы батча остаются вместе и идут в исходном порядке: "
            "`limit` считает запросы, а не отдельные отзывы. "
            "`request_id` совпадает с заголовком `X-Request-Id` ответа `/predict`.\n\n"
            "`review_text` содержит текст, если во время прогноза было включено "
            "`STORE_REVIEW_TEXT=1`. Иначе возвращается `null`, "
            "включение настройки не восстанавливает тексты старых записей. "
        ),
        response_description="Сохранённые запросы и их прогнозы.",
        responses={
            200: {
                "content": {
                    "application/json": {
                        "examples": {
                            "history": {
                                "summary": "История запросов от новых к старым",
                                "value": [
                                    {
                                        "request_id": "a5c762ff96b945b7a4d619e530f026e8",
                                        "created_at": "2026-10-08T19:59:55+00:00",
                                        "model_id": "tfidf",
                                        "model_version": "2026-10-07T08:28:11+00:00",
                                        "item_count": 1,
                                        "inference_ms": 12.5,
                                        "predictions": [
                                            {
                                                "item_index": 0,
                                                "dataset_id": 1,
                                                "review_text": "Excellent service!",
                                                "label": 5,
                                                "confidence": 0.98,
                                            }
                                        ],
                                    },
                                    {
                                        "request_id": "b7d026e862a94ea8a5c761ff96b945c3",
                                        "created_at": "2026-10-08T18:00:00+00:00",
                                        "model_id": "deberta",
                                        "model_version": "full-28e21afd362a",
                                        "item_count": 1,
                                        "inference_ms": 217.0,
                                        "predictions": [
                                            {
                                                "item_index": 0,
                                                "dataset_id": 2,
                                                "review_text": "My order never arrived.",
                                                "label": 1,
                                                "confidence": 0.91,
                                            }
                                        ],
                                    },
                                ],
                            },
                            "empty": {
                                "summary": "История пуста или записи закончились",
                                "value": [],
                            },
                        }
                    }
                }
            },
            422: {
                "model": ValidationErrorResponse,
                "description": "Неверный тип или значение limit/offset.",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": [
                                {
                                    "loc": ["query", "limit"],
                                    "message": "Input should be less than or equal to 100",
                                }
                            ]
                        }
                    }
                },
            },
            503: {
                "model": ErrorResponse,
                "description": "Не удалось прочитать SQLite. Попробуйте повторить запрос.",
                "content": {
                    "application/json": {
                        "example": {"detail": "История временно недоступна"}
                    }
                },
            },
        },
    )
    def history(
        limit: Annotated[
            int,
            Query(
                ge=1,
                le=100,
                description="Число запросов на странице, включая все прогнозы каждого.",
                examples=[10],
            ),
        ] = 10,
        offset: Annotated[
            int,
            Query(
                ge=0,
                le=2**63 - 1,
                description="Сколько новых запросов пропустить, 0 — начать с самого нового.",
                openapi_examples={
                    "first_page": {"summary": "Первая страница", "value": 0},
                    "next_page": {"summary": "Следующая страница при limit=10", "value": 10},
                },
            ),
        ] = 0,
    ):
        try:
            return application.state.history.read(limit=limit, offset=offset)
        except (sqlite3.Error, OSError):
            LOGGER.exception("Request history could not be read")
            raise HTTPException(503, "История временно недоступна") from None

    @application.post(
        "/load_model",
        response_model=ModelState,
        summary="Переключить активную модель",
        tags=["Модели"],
        description=(
            "Передайте `model_id` из `GET /models`. Выбор действует для всех последующих "
            "запросов к сервису. Уже начатые прогнозы завершатся с прежней моделью.\n\n"
            "Переключение выполняется после проверки файлов и загрузки модели, "
            "это может занять несколько секунд. При ошибке прежняя модель остаётся активной. "
            "После перезапуска сервиса снова выбирается `tfidf`."
        ),
        response_description="Идентификатор и версия выбранной модели.",
        responses={
            404: {
                "model": ErrorResponse,
                "description": "Модель с таким model_id не зарегистрирована.",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "Неизвестная модель, выберите model_id из GET /models"
                        }
                    }
                },
            },
            422: {
                "model": ValidationErrorResponse,
                "description": "Отсутствует model_id или нарушен формат запроса.",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": [
                                {
                                    "loc": ["body", "model_id"],
                                    "message": "Field required",
                                }
                            ]
                        }
                    }
                },
            },
            503: {
                "model": ErrorResponse,
                "description": "Не удалось загрузить модель. Прежняя остаётся активной.",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "Модель не прошла проверку, предыдущая модель остаётся активной"
                        }
                    }
                },
            },
        },
    )
    def load_model(
        selection: Annotated[
            ModelSelection,
            Body(
                openapi_examples={
                    "tfidf": {
                        "summary": "TF-IDF: слова, символы и числовые признаки",
                        "value": {"model_id": "tfidf"},
                    },
                    "word-only": {
                        "summary": "TF-IDF: слова и биграммы",
                        "value": {"model_id": "word-only"},
                    },
                    "deberta": {"summary": "DeBERTa", "value": {"model_id": "deberta"}},
                }
            ),
        ],
    ):
        try:
            active = application.state.registry.load(selection.model_id)
        except UnknownModelError:
            raise HTTPException(
                404, "Неизвестная модель, выберите model_id из GET /models"
            ) from None
        except ModelLoadError:
            LOGGER.exception("Registered model loading failed")
            raise HTTPException(
                503, "Модель не прошла проверку, предыдущая модель остаётся активной"
            ) from None
        return {"model_id": active.spec.id, "model_version": active.spec.version}

    @application.post(
        "/predict",
        response_model=Prediction | list[Prediction],
        summary="Предсказать оценку отзыва",
        tags=["Прогноз"],
        description=(
            "Передайте объект с полем `Review` или массив таких объектов. `Id` необязателен. "
            "Один отзыв возвращает один объект, массив — массив ответов в исходном порядке.\n\n"
            "**Ограничения**\n\n"
            "- `Review`: англоязычный текст от 1 до 20 000 символов, не только пробелы.\n"
            "- Массив: от 1 до 128 отзывов, суммарно до 200 000 символов текста.\n"
            "- Дополнительные поля запрещены.\n"
            "- DeBERTa использует первые 128 токенов отзыва, включая служебные.\n\n"
            "**Результат**\n\n"
            "`label` — оценка 1–5, выбранная как медиана распределения для метрики MAE. "
            "`confidence` — оценённая моделью вероятность выбранной оценки. \n\n"
            "Обработанный запрос и ответы сохраняются в SQLite, тексты отзывов по умолчанию "
            "не записываются. Просмотр истории — GET /history. "
            "Заголовки ответа содержат модель, её версию и идентификатор запроса."
        ),
        response_description="Оценка и её вероятность для каждого переданного отзыва.",
        responses={
            200: {
                "headers": {
                    "X-Model-Id": {
                        "description": "Модель, выполнившая прогноз.",
                        "schema": {"type": "string"},
                    },
                    "X-Model-Version": {
                        "description": "Версия файлов модели.",
                        "schema": {"type": "string"},
                    },
                    "X-Request-Id": {
                        "description": "Идентификатор запроса в истории SQLite.",
                        "schema": {"type": "string"},
                    },
                },
                "content": {
                    "application/json": {
                        "examples": {
                            "single": {
                                "summary": "Один отзыв",
                                "value": {"label": 5, "confidence": 0.98},
                            },
                            "batch": {
                                "summary": "Два отзыва",
                                "value": [
                                    {"label": 5, "confidence": 0.98},
                                    {"label": 1, "confidence": 0.91},
                                ],
                            },
                        }
                    }
                },
            },
            422: {
                "model": ValidationErrorResponse | ErrorResponse,
                "description": "Некорректные поля, JSON или превышение ограничений. Прогноз не выполнен.",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "Суммарная длина Review превышает 200 000 символов"
                        }
                    }
                },
            },
            503: {
                "model": ErrorResponse,
                "description": "Не удалось выполнить прогноз или сохранить историю запроса.",
                "content": {
                    "application/json": {
                        "examples": {
                            "model": {
                                "summary": "Ошибка прогнозирования",
                                "value": {
                                    "detail": "Модель не смогла обработать запрос"
                                },
                            },
                            "history": {
                                "summary": "История недоступна",
                                "value": {
                                    "detail": "История временно недоступна, повторите запрос позже"
                                },
                            },
                        }
                    }
                },
            },
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
                503, "История временно недоступна, повторите запрос позже"
            ) from None
        response.headers["X-Model-Id"] = active.spec.id
        response.headers["X-Model-Version"] = active.spec.version
        response.headers["X-Request-Id"] = request_id
        return predictions[0] if single else predictions

    return application


app = create_app()
