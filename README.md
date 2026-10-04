# Sentiment Analysis of Company Reviews

Прогноз оценки компании от **1 до 5** по англоязычному отзыву. Решение тестового задания на данных [соревнования Kaggle](https://www.kaggle.com/competitions/sentiment-analysis-company-reviews): EDA, сравнение моделей и Docker API с историей запросов в SQLite.

## Запуск

Нужен работающий Docker с Compose v2. После клонирования, из корня проекта:

```bash
docker compose up --build -d --wait
```

Обученные модели включены в репозиторий: данные Kaggle, Python и ручная загрузка весов для запуска сервиса не нужны. Первая сборка скачивает базовый образ и зависимости.

- Swagger: <http://localhost:8000/docs>
- Готовность: <http://localhost:8000/health>
- Остановка: `docker compose down` — история сохраняется в Docker volume.

## API

```bash
curl http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"Id": 1, "Review": "Excellent service and fast delivery!"}'
```

Ответ содержит `label` — целую оценку 1–5 — и `confidence` от 0 до 1. Это вероятность **возвращённой оценки**, а не гарантия правильности. Оценка выбирается как медиана распределения вероятностей для MAE.

Вместо объекта можно передать массив объектов; ответ будет массивом в том же порядке. `Id` необязателен и не влияет на прогноз. Ограничения: до 128 отзывов, до 20 000 символов в отзыве, суммарно до 200 000 символов и до 1 МиБ JSON. Неверные поля возвращают `422`, слишком большое тело — `413`.

```bash
curl http://localhost:8000/models
curl http://localhost:8000/load_model \
  -H 'Content-Type: application/json' -d '{"model_id": "word-only"}'
```

Доступны `tfidf` (по умолчанию) и `word-only`. Для возврата основной модели передайте `{"model_id":"tfidf"}`. Смена проверяет файл и пробный прогноз; ошибка сохраняет предыдущую модель. После перезапуска снова выбирается `tfidf`.

Каждый успешный запрос `/predict` записывается в SQLite вместе с версией модели. Заголовки `X-Model-Id`, `X-Model-Version`, `X-Request-Id` связывают ответ с историей. По умолчанию сохраняются хеш и длина отзыва; хранение самого текста включается через `STORE_REVIEW_TEXT=1 docker compose up -d`.

## Качество

После удаления 24 повторов осталось 59 976 отзывов. Сравнение ниже использует одинаковые пять `StratifiedGroupKFold`-разбиений: тексты, совпадающие после изменения регистра и пробелов, всегда находятся вместе. Меньшая MAE лучше.

| Модель | Средняя MAE пяти фолдов |
|---|---:|
| Постоянная медиана обучающих оценок | 1,43781 |
| Слова TF-IDF + Logistic Regression | 0,19008 |
| **Слова + символы + 13 числовых признаков** | **0,18539** |

Итоговые параметры выбраны через `GridSearchCV`: 9 сочетаний `C` и `min_df`, по 5 фолдов. Эти же данные использовались при выборе признаков, поэтому таблица показывает качество при подборе модели; независимого тестового результата или Kaggle score здесь нет. Числа и параметры сохранены в [training_metrics.json](models/training_metrics.json).

Отдельный эксперимент на первом фолде дал DeBERTa **0,14722**, MPNet + CatBoost — **0,44325**. Сопоставление на тех же строках и причины выбора компактного TF-IDF для сервиса — в [отчёте о решениях](docs/decisions.md).

На Apple M1 в Linux Docker (2 CPU, 2 ГБ) p95 одиночного HTTP-запроса после прогрева — **до 10,4 мс** для трёх проверенных длин, пакета из 16 — **22,2 мс**, включая SQLite. [Измерения](reports/http_benchmark.json) и [проверка чистого клона](reports/verification.json) сохранены. **41 автоматический тест прошёл.**

## Как читать проект

1. [eda.ipynb](notebooks/eda.ipynb) — данные, повторы, распределение оценок, гипотезы о признаках.
2. [baselinev2.ipynb](notebooks/baselinev2.ipynb) — основной ноутбук: медиана, признаки, сравнения, GridSearchCV, итоговая модель.
3. [mpnet_catboost.ipynb](notebooks/mpnet_catboost.ipynb), [deberta.ipynb](notebooks/deberta.ipynb) — дополнительные, более тяжёлые эксперименты.
4. [docs/decisions.md](docs/decisions.md) — обоснования, источники, ограничения и устройство сервиса.

Код обучения и API — в `company_reviews/`; артефакты и метрики — в `models/`; автоматические проверки — в `tests/`.

## Воспроизведение и проверки

Для разработки нужен [uv](https://docs.astral.sh/uv/). Команды выполняются из корня проекта:

```bash
uv sync --locked --extra api --extra eda --group dev
uv run --locked pytest -q
uv run --locked ruff check company_reviews tests scripts
uv run --locked python scripts/smoke_api.py
```

Последняя команда проверяет уже запущенный Docker API. Для повторного обучения скачайте `train.csv`, `test.csv`, `sample_submission.csv` со страницы соревнования в `data/raw/`. Доступ к данным регулируется правилами Kaggle; CSV и кэши не включены в Git.

```bash
uv run --locked python -m company_reviews.training --tune
uv run --locked python -m company_reviews.benchmark
```

Обучение заново создаёт обе модели и метрики. Ноутбуки используют ядро из `.venv`; для MPNet и DeBERTa дополнительно установите `uv sync --locked --extra api --extra eda --extra advanced --group dev`. Время и ресурсы запусков описаны в отчёте. GitHub Actions настроен на тесты, сборку Docker и HTTP smoke test.

## Использование ИИ

Проект разрабатывался с помощью OpenAI Codex: исследование решений, подготовка кода и документации, запуск экспериментов и отдельные агентские ревью. Результаты в таблицах получены выполнением кода; проверки и ограничения описаны в отчёте.
