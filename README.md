# Akkadian → English Neural Machine Translator

**Авторы:** Ангелина Черникова и Агафонов Руслан

**Группа:** 972401

## Описание

Нейросетевой переводчик с аккадского языка (транслитерация) на английский, основанный на модели ByT5-small (byte-level T5). Аккадский — язык древней Месопотамии, один из наименее ресурсных языков для задач машинного перевода.

## Архитектура

- **Модель:** `google/byt5-small` (байтовый уровень, обрабатывает диакритику š/ṭ/ā без потерь)
- **Инференс:** FastAPI + SSE стриминг
- **Трекинг:** Weights & Biases

## Результаты (Ablation Table)

| Technique | BLEU on dev | chrF++ on dev | Notes |
|-----------|------------|--------------|-------|
| Baseline (no norm, greedy, beam=1) | 5.13 | 19.88 | ByT5-small, 5 epochs, 1061 train samples |
| + Orthography normalization | 1.18 | 14.77 | Нормализация снижает качество — диакритика несёт смысловую нагрузку |
| Beam search beam=4 | 4.49 | 19.83 | Незначительно уступает greedy на малом датасете |
| Mini-ensemble | planned | planned | 2 checkpoint с разными seed — запланировано |

**Вывод:** Для данного малоресурсного датасета (1061 пар) greedy decoding без нормализации даёт лучший chrF++ (19.88). Orthography normalization неожиданно снижает качество — ByT5 работает на байтовом уровне и диакритика (š, ṭ, ā, ū) несёт важную фонетическую информацию.

## Метрики на dev-выборке (100 примеров)

| Метрика | Значение |
|---------|---------|
| BLEU | 5.13 |
| chrF++ | 19.88 |
| COMET | N/A (модель не обучена на аккадском) |
| TTFT (медиана) | ~5 сек (CPU) |
| Tokens/sec | ~2-3 tok/s (CPU) |

## Датасеты

| Датасет | Размер | Лицензия |
|---------|--------|----------|
| Kaggle Deep Past Initiative — train.csv | 1561 строк | Соревновательные данные |
| Dev split (последние 500 строк train) | 500 строк | — |
| Test set | 4 строки | Соревновательные данные |


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

### Интерфейс
![Akkadian Translator UI](img/web.png)


## Ресурсы

- [ByT5 paper](https://arxiv.org/abs/2105.13626)
- [Gutherz et al. 2023](https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349)
- [ORACC corpus](http://oracc.museum.upenn.edu/)
- [Kaggle competition](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)