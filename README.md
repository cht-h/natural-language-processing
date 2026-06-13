# Akkadian → English Neural Machine Translator

**Автор:** Ангелина Черникова и Агафонов Руслан
**Группа:** 972401

## Описание

Нейросетевой переводчик с аккадского языка (транслитерация) на английский, основанный на модели ByT5-small (byte-level T5). Аккадский — язык древней Месопотамии, один из наименее ресурсных языков для задач машинного перевода.

## Архитектура

- **Модель:** `google/byt5-small` (байтовый уровень, обрабатывает диакритику š/ṭ/ā без потерь)
- **Инференс:** FastAPI + SSE стриминг
- **Трекинг:** Weights & Biases

## Результаты

| Technique | chrF++ on dev | Notes |
|-----------|--------------|-------|
| Baseline (greedy, no norm) | 14.77 | ByT5-small, 5 epochs, 100 dev samples |
| + Orthography normalization | TBD | |
| + Beam search (beam=4) | 14.77 | beam=4 использован в baseline |
| + Ensemble (2 checkpoints) | TBD | |

## Датасеты

| Датасет | Размер | Лицензия |
|---------|--------|----------|
| Kaggle Deep Past Initiative (train.csv) | 1561 строк | Соревновательные данные |
| Dev split (последние 500 строк train) | 500 строк | — |

## Установка и запуск

### Локально

```bash
# Клонировать репозиторий
git clone https://github.com/cht-h/natural-language-processing
cd natural-language-processing

# Активировать окружение
& "C:\Users\...\Scripts\activate.ps1"

# Установить зависимости
pip install -r requirements.txt

# Обучить модель
python model.py train --dataset_path=data/raw/train.csv

# Перевести текст
python model.py predict --text="sarrum ana alim illik"

# Предсказать файл
python model.py predict_file --dataset_path=data/raw/test.csv
```

### Docker

```bash
# CPU профиль
docker compose --profile cpu up

# GPU профиль  
docker compose --profile gpu up

# Открыть в браузере
open http://localhost:8000
```

## Веб-приложение

После запуска сервера открой `http://localhost:8000` — появится интерфейс переводчика с реальным стримингом токенов.

## Метрики

| Метрика | Dev (100 samples) |
|---------|-------------------|
| BLEU | 1.18 |
| chrF++ | 14.77 |
| COMET | N/A |

## Ресурсы

- [ByT5 paper](https://arxiv.org/abs/2105.13626)
- [Gutherz et al. 2023](https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349)
- [ORACC corpus](http://oracc.museum.upenn.edu/)
- [Kaggle competition](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)