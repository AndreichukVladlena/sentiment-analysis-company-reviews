# Финальная DeBERTa через Git LFS

Финальная модель поставляется в `models/deberta/` вместе с проектом. Большой файл `model.safetensors` хранится в Git LFS, а токенизатор, конфигурации и контрольные суммы в `models/manifest.json` — в обычном Git. Отдельный архив и локальный кэш автора не нужны.

## Получение готовой модели

Нужны Git, установленный [Git LFS](https://git-lfs.com/) и работающий Docker с Compose. Python и PyTorch на хосте для запуска контейнера не требуются.

```bash
git clone https://github.com/AndreichukVladlena/sentiment-analysis-company-reviews.git
cd sentiment-analysis-company-reviews
git lfs install --local
git lfs pull --include="models/deberta/model.safetensors"
git lfs fsck
```

При уже настроенном Git LFS веса обычно загружаются при клонировании. Явный `git lfs pull` также подходит для ранее клонированного проекта после `git pull`. Не продолжайте сборку DeBERTa, если скачивание или проверка завершились ошибкой.

Без LFS вместо весов может остаться небольшой текстовый файл, начинающийся с `version https://git-lfs.github.com/spec/v1`. Это указатель, а не модель. Выполните команды LFS выше; одного обычного `git pull` недостаточно, если загрузка больших файлов была пропущена. Предпочитайте `git clone` скачиванию ZIP: включение LFS-объектов в архивы зависит от настроек GitHub.

## Состав и проверка

Каталог содержит семь файлов: `model.safetensors`, `config.json`, `spm.model`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `training_config.json`. Данные Kaggle и модели пяти фолдов в поставку не входят. Каталог занимает около 740 МБ.

Это отдельная модель, обученная на всех **59 976** очищенных отзывах: `training_scope="full"`, максимум 128 токенов и исходный медленный токенизатор. SHA256 поставляемого файла весов:

```text
28e21afd362a6fe6ca3cca7d5a80e705e2d30c7168802c6f72ee66b30c9315d9
```

На macOS проверьте его командой `shasum -a 256 models/deberta/model.safetensors`, на Linux — `sha256sum models/deberta/model.safetensors`. Сервис дополнительно сверяет контрольные суммы всех семи файлов при загрузке модели.

## Запуск API

Из корня проекта после успешного получения весов:

```bash
MODEL_EXTRA=deberta docker compose up --build -d --wait
curl http://localhost:8000/load_model \
  -H 'Content-Type: application/json' -d '{"model_id":"deberta"}'
curl http://localhost:8000/predict \
  -H 'Content-Type: application/json' -d '{"Review":"Excellent service and fast delivery!"}'
```

`MODEL_EXTRA=deberta` добавляет зависимости DeBERTa в образ; используйте переменную при каждой его пересборке. Все файлы модели копируются в образ. Во время предсказаний доступ к Hugging Face не нужен; первая сборка требует доступа к реестрам образов и Python-пакетов.

TF-IDF остаётся моделью по умолчанию, в том числе после рестарта. Для DeBERTa повторно вызовите `/load_model`. Наличие записи `deberta` в `/models` означает регистрацию модели, но не подтверждает получение её весов и установку зависимостей. Если загрузка не удалась, предыдущая модель остаётся активной.

## Запуск только baseline

Две компактные TF-IDF-модели находятся в обычном Git. Если DeBERTa не нужна, можно пропустить скачивание её весов:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/AndreichukVladlena/sentiment-analysis-company-reviews.git
cd sentiment-analysis-company-reviews
docker compose up --build -d --wait
```

Для последующего подключения DeBERTa выполните получение весов и пересборку из разделов выше. Исходные данные и повторное обучение для готовой поставки не нужны.

## Публикация новых весов

Этот раздел нужен только после собственного финального обучения. Экспорт использует стандартную библиотеку Python и принимает полный совместимый checkpoint:

```bash
python3 -m company_reviews.export_deberta /path/to/full-checkpoint
git lfs install --local
git add models/deberta models/manifest.json
git diff --cached --stat
git show :models/deberta/model.safetensors
git lfs status
git commit -m "feat: update final DeBERTa weights"
git lfs fsck
git push origin main
```

Перед коммитом `git show` должен вывести короткий LFS-указатель, а не бинарные данные. Правило в `.gitattributes` применяется только к финальному файлу весов. Не добавляйте `data/cache/` и модели проверочных фолдов. Коммитьте веса, токенизатор, конфигурации и manifest вместе; при изменении весов обновите также контрольную сумму в этой инструкции.

Обычный `git push` с установленным LFS использует pre-push hook для загрузки больших файлов. Не отключайте этот hook. Каждая новая версия весов занимает дополнительное место, а скачивания расходуют LFS-трафик владельца репозитория. Перед публикацией проверьте [квоты и бюджет Git LFS](https://docs.github.com/en/billing/concepts/product-billing/git-lfs); при исчерпании квоты скачивание может стать недоступным.

Кросс-валидация оценивает выбранный подход; отдельной независимой оценки качества финальной модели нет.
