# Sentiment Analysis of Company Reviews

Прогноз оценки компании **от 1 до 5** по англоязычному отзыву. Тестовое задание на данных [Kaggle Company Reviews](https://www.kaggle.com/competitions/sentiment-analysis-company-reviews): исследование данных, сравнение моделей и API в Docker.

## Запуск

Нужны Docker с Compose и [Git LFS](https://git-lfs.com/). Команды выполняйте в терминале.

### 1. Склонируйте репозиторий и загрузите веса DeBERTa

```bash
git clone https://github.com/AndreichukVladlena/sentiment-analysis-company-reviews.git
cd sentiment-analysis-company-reviews
git lfs install --local
git lfs pull --include="models/deberta/model.safetensors"
git lfs fsck
```

### 2. Соберите и запустите сервис

```bash
MODEL_EXTRA=deberta docker compose up --build -d --wait
```

Эта сборка поддерживает TF-IDF и DeBERTa. Необходимые библиотеки устанавливаются внутри Docker. При последующих пересборках используйте эту же команду.

### 3. Выберите модель

После запуска активна `tfidf`. Чтобы переключиться на DeBERTa:

```bash
curl http://localhost:8000/load_model \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "deberta"}'
```

### Вариант только с TF-IDF

Если нужна только TF-IDF, вместо шагов 1–3 выполните:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/AndreichukVladlena/sentiment-analysis-company-reviews.git
cd sentiment-analysis-company-reviews
docker compose up --build -d --wait
```

Эта сборка запускает сервис с TF-IDF без скачивания весов и установки зависимостей DeBERTa.

### Swagger : <http://localhost:8000/docs>

Сервис сохраняет журнал обработанных запросов в SQLite: время, использованную модель и результаты прогнозирования. Тексты отзывов по умолчанию не сохраняются, их хранение можно включить через `STORE_REVIEW_TEXT=1 docker compose up -d`.

## Результаты

После удаления точных повторов осталось 59 976 отзывов. Baseline и DeBERTa сравнивались на одинаковых пяти фолдах. Совпадающие после изменения регистра и пробелов отзывы не разделяются между обучением и проверкой.

| Модель | Средняя MAE пяти фолдов |
| --- | ---: |
| Постоянная медиана | 1,43781 |
| TF-IDF по словам и биграммам + Logistic Regression | 0,19008 |
| TF-IDF по словам и символам + 13 числовых признаков + Logistic Regression | 0,18507 |
| **DeBERTa-v3-base: два верхних слоя + голова, одна эпоха** | **0,14371** |

Финальный baseline использует `C=4`, словный `min_df=2` и вес числовых признаков `0,05`. DeBERTa уменьшает среднюю MAE на **0,04137 (22,35%)** относительно этого варианта. Стандартное отклонение пяти MAE DeBERTa — 0,00374 (`ddof=0`); общая OOF MAE — 0,143707483. Ошибки DeBERTa на редких оценках 2–4 остаются заметно выше, чем на 1 и 5.

### Результаты Kaggle

Baseline (TF-IDF по словам и символам + 13 числовых признаков + Logistic Regression) и DeBERTa, обученные на всех 59 976 очищенных отзывах, отправлены в [Sentiment Analysis — Company Reviews](https://www.kaggle.com/competitions/sentiment-analysis-company-reviews). Обе отправки успешно завершены со статусом `Complete (after deadline)`.

![Результаты финальных DeBERTa и TF-IDF на Kaggle: обе отправки успешно завершены, показаны Private и Public MAE](docs/images/kaggle-results-2026-10-07.png)

Для обеих моделей оценка выбиралась как медиана предсказанного распределения.

## Структура проекта

| Файл или папка | Содержание |
| --- | --- |
| `notebooks/` | Анализ данных, эксперименты с признаками, подбор параметров и обучение моделей TF-IDF + Logistic Regression и DeBERTa |
| [docs/decisions.md](docs/decisions.md) | Обоснования выбора и результаты экспериментов |
| `company_reviews/` | Признаки, обучение, экспорт моделей и сервис |
| `models/` | Две TF-IDF-модели, финальная DeBERTa (веса через LFS) и каталог для `/load_model` |

## Использование ИИ

При подготовке кода, исследовании решений и проверках использовался OpenAI Codex.
