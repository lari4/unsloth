# Agent Pipelines Documentation

Этот документ описывает все возможные схемы работы агента в приложении Unsloth, включая последовательность вызовов промптов и обработку данных.

## Оглавление

1. [Synthetic Data Generation Pipeline](#synthetic-data-generation-pipeline)
2. [AIME Evaluation Pipeline](#aime-evaluation-pipeline)
3. [Training Pipeline (GSM8K/LIMO)](#training-pipeline-gsm8klimo)
4. [Vision OCR Pipeline](#vision-ocr-pipeline)

---

## Synthetic Data Generation Pipeline

### Описание
Пайплайн для генерации синтетических данных из различных источников (PDF, HTML, YouTube, DOCX, PPT, TXT). Создает QA пары для обучения LLM моделей с автоматической оценкой качества.

### Компоненты

**Основной класс:** `SyntheticDataKit`
**Расположение:** `unsloth/dataprep/synthetic.py`
**Конфигурация:** `unsloth/dataprep/synthetic_configs.py`

### Схема работы

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYNTHETIC DATA GENERATION PIPELINE                │
└─────────────────────────────────────────────────────────────────────┘

1. ИНИЦИАЛИЗАЦИЯ
   ┌──────────────────┐
   │ SyntheticDataKit │
   │  .from_pretrained│
   └────────┬─────────┘
            │
            ├─> Загрузка модели (vLLM server)
            │   - model_name (default: Llama-3.1-8B-Instruct-unsloth-bnb-4bit)
            │   - max_seq_length: 2048
            │   - gpu_memory_utilization: 0.98
            │
            └─> Запуск vLLM subprocess
                ├─> Ожидание готовности сервера (timeout: 1200s)
                └─> Проверка статуса: http://localhost:8000/metrics

2. ПОДГОТОВКА
   ┌────────────────────────┐
   │ prepare_qa_generation  │
   └────────┬───────────────┘
            │
            ├─> Создание структуры папок
            │   - pdf, html, youtube, docx, ppt, txt
            │   - output, generated, cleaned, final
            │
            └─> Генерация конфигурации (synthetic_data_kit_config.yaml)
                - температура: 0.7
                - top_p: 0.95
                - max_tokens: 512
                - num_pairs: 25
                - cleanup_threshold: 1.0

3. CHUNKING (разбиение текста)
   ┌──────────────┐
   │  chunk_data  │
   └──────┬───────┘
          │
          ├─> Чтение входного файла
          │
          ├─> Токенизация
          │   max_tokens = max_seq_length - max_generation_tokens*2 - 128
          │
          ├─> Расчет границ chunks с overlap
          │   n_chunks = ceil(length / (max_tokens - overlap))
          │
          └─> Сохранение chunks в отдельные файлы
              filename_0.txt, filename_1.txt, ...

4. ГЕНЕРАЦИЯ QA ПАР
   ┌─────────────────────────┐
   │ API Call: /v1/completions│
   └──────┬──────────────────┘
          │
          ├─> ПРОМПТ 1: Summary Generation (опционально)
          │   └─> Вход: document text
          │   └─> Выход: 3-5 sentence summary
          │
          └─> ПРОМПТ 2: QA Generation
              ├─> Вход:
              │   - text: chunked document
              │   - num_pairs: 25
              │   - temperature: 0.7
              │   - top_p: 0.95
              │   - max_tokens: 512
              │
              └─> Выход: JSON array of QA pairs
                  [{"question": "...", "answer": "..."}]

5. ОЦЕНКА КАЧЕСТВА
   ┌─────────────────────────┐
   │ API Call: /v1/completions│
   └──────┬──────────────────┘
          │
          └─> ПРОМПТ 3: QA Rating
              ├─> Вход:
              │   - pairs: сгенерированные QA пары
              │   - batch_size: 4
              │   - temperature: 0.3 (ниже для консистентности)
              │
              └─> Выход: JSON array with ratings
                  [{"question": "...", "answer": "...", "rating": 8}]

6. ФИЛЬТРАЦИЯ И СОХРАНЕНИЕ
   ┌───────────────┐
   │   Cleanup     │
   └───────┬───────┘
           │
           ├─> Фильтрация по threshold (1-10 scale)
           │   Сохраняются только пары с rating >= threshold
           │
           ├─> Сохранение в cleaned/
           │
           └─> Конвертация в финальный формат
               └─> Сохранение в final/
                   - JSONL format (default)
                   - включение metadata (опционально)

7. ЗАВЕРШЕНИЕ
   ┌──────────┐
   │ cleanup  │
   └────┬─────┘
        │
        ├─> Остановка vLLM сервера
        │   - graceful termination (10s timeout)
        │   - force kill if needed
        │
        ├─> Очистка CUDA cache
        │   torch.cuda.empty_cache()
        │
        └─> Удаление vLLM модуля
```

### Поток данных

```
INPUT                    PROCESS                    OUTPUT
─────                    ───────                    ──────

Document.pdf    ──>  Chunking        ──>  document_0.txt
                     (overlap=64)          document_1.txt
                                          document_2.txt
                            │
                            ├──>  QA Generation   ──>  Raw QA pairs
                            │     (25 pairs/chunk)      (JSON)
                            │
                            └──>  QA Rating       ──>  Rated QA pairs
                                  (1-10 scale)          (JSON with ratings)
                                       │
                                       └──>  Filtering   ──>  High-quality QA
                                             (threshold)       (JSONL)
```

### Ключевые параметры конфигурации

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| temperature | 0.7 | Креативность генерации QA |
| top_p | 0.95 | Nucleus sampling |
| chunk_size | max_seq - 2*max_gen - 2 | Размер текстовых chunks |
| overlap | 64 | Перекрытие между chunks |
| max_tokens | 512 | Макс длина генерируемого ответа |
| num_pairs | 25 | QA пар на chunk |
| cleanup_threshold | 1.0 | Минимальный рейтинг для сохранения |
| cleanup_batch_size | 4 | Размер батча для рейтинга |
| cleanup_temperature | 0.3 | Температура для рейтинга |

### API Endpoints

- **Server URL:** `http://localhost:8000/v1`
- **Metrics:** `http://localhost:8000/metrics`
- **Completions:** `http://localhost:8000/v1/completions`

---

