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

## AIME Evaluation Pipeline

### Описание
Пайплайн для оценки языковых моделей на комбинированном датасете AIME (American Invitational Mathematics Examination). Включает test2024, test2025-I и test2025-II. Использует Pass@k метрику с множественной генерацией.

### Компоненты

**Основная функция:** `evaluate_model_aime`
**Расположение:** `tests/utils/aime_eval.py`
**Вспомогательные функции:** `load_aime_dataset`, `extract_aime_answer`, `compare_aime_results`

### Схема работы

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AIME EVALUATION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

1. ЗАГРУЗКА ДАТАСЕТА
   ┌────────────────────────────────┐
   │ download_and_combine_datasets  │
   └──────────┬─────────────────────┘
              │
              ├─> Скачивание трех датасетов:
              │   ├─> test2024.jsonl
              │   ├─> test2025-I.jsonl
              │   └─> test2025-II.jsonl
              │
              ├─> Комбинирование в один файл
              │   └─> data/aime/aime.jsonl
              │
              └─> Добавление метаданных:
                  - source_dataset
                  - original_id
                  - global_id

2. ФОРМАТИРОВАНИЕ ПРОМПТОВ
   ┌──────────────────┐
   │ load_aime_dataset│
   └────────┬─────────┘
            │
            └─> Создание структуры для каждой задачи:
                {
                  "global_id": N,
                  "source_dataset": "test2024",
                  "problem": "текст задачи",
                  "answer": "123",
                  "prompt": [
                    {
                      "role": "system",
                      "content": "You are a mathematical problem solver..."
                    },
                    {
                      "role": "user",
                      "content": "Problem: ... Solve this step by step..."
                    }
                  ]
                }

3. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
   ┌──────────────────────────┐
   │ FastLanguageModel.from_  │
   │     pretrained (vLLM)    │
   └──────────┬───────────────┘
              │
              ├─> Конфигурация:
              │   - fast_inference: True (vLLM)
              │   - gpu_memory_utilization: 0.8
              │   - max_seq_length: variable
              │
              └─> Sampling params:
                  - temperature: 0.3
                  - top_p: 0.95
                  - max_tokens: 32768
                  - n: 8 (множественная генерация)
                  - seed: 0

4. ГЕНЕРАЦИЯ РЕШЕНИЙ (для каждой задачи)
   ┌─────────────────────────┐
   │ model.fast_generate     │
   └──────┬──────────────────┘
          │
          ├─> Применение chat template
          │   prompt_text = tokenizer.apply_chat_template(
          │       item["prompt"],
          │       add_generation_prompt=True
          │   )
          │
          ├─> Подсчет входных токенов
          │
          ├─> Генерация N=8 ответов
          │   └─> Параллельная генерация с одинаковыми параметрами
          │
          └─> Извлечение ответов
              outputs = [output.text for output in outputs]

5. ИЗВЛЕЧЕНИЕ ОТВЕТОВ
   ┌─────────────────────┐
   │ extract_aime_answer │
   └──────┬──────────────┘
          │
          ├─> Поиск паттернов (в порядке приоритета):
          │   1. "the answer is 123"
          │   2. "therefore, the answer is 123"
          │   3. "\boxed{123}"
          │   4. "$\boxed{123}$"
          │   5. "answer: 123"
          │   6. standalone number "123"
          │
          ├─> Валидация (0 <= number <= 999)
          │
          └─> Возврат последнего найденного числа

6. ОЦЕНКА РЕЗУЛЬТАТОВ
   ┌──────────────────┐
   │ Scoring Logic    │
   └────┬─────────────┘
        │
        ├─> Для каждой задачи:
        │   ├─> Извлечение всех N=8 ответов
        │   ├─> Сравнение с ground truth
        │   └─> is_correct = any(answer == ground_truth)
        │
        ├─> Расчет метрик:
        │   ├─> Accuracy = correct / total * 100
        │   └─> Pass@k = (задачи с >= 1 правильным) / total * 100
        │
        └─> Разбивка по датасетам:
            - test2024 accuracy
            - test2025-I accuracy
            - test2025-II accuracy

7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
   ┌─────────────────┐
   │ Save to JSON    │
   └────┬────────────┘
        │
        └─> Файл: aime_eval_combined_{model_type}_t{temp}_n{n}.json
            {
              "results": {
                "accuracy": 45.2,
                "pass_at_k": 52.1,
                "source_accuracies": {...},
                "avg_input_tokens": 234,
                "avg_output_tokens": 512
              },
              "records": {
                "0": {
                  "problem": "...",
                  "ground_truth": "123",
                  "responses": [...],
                  "extracted_answers": [...],
                  "is_correct": true,
                  "n_correct": 3,
                  "n_total": 8
                }
              }
            }
```

### Поток данных

```
ИСТОЧНИК                 ОБРАБОТКА              РЕЗУЛЬТАТ
────────                 ─────────              ─────────

GitHub JSONL    ──>  Download &        ──>  Combined Dataset
(3 файла)            Combine                 (45+ problems)
                         │
                         ├──>  Format Prompts    ──>  Structured Data
                         │     (system + user)
                         │
                         └──>  Generate (N=8)    ──>  Multiple Solutions
                                    │                  per Problem
                                    │
                                    ├──>  Extract        ──>  Numerical Answers
                                    │     Answers             (0-999)
                                    │
                                    └──>  Evaluate       ──>  Metrics + Records
                                          Pass@k              (JSON)
```

### Метрики

**Accuracy:**
```
Accuracy = (Количество задач с >= 1 правильным ответом) / Всего задач * 100
```

**Pass@k:**
```
Pass@k = Вероятность того, что хотя бы один из k ответов правильный
       = count(задачи где n_correct > 0) / total_problems * 100
```

**Performance Tiers:**
- 🏆 EXCEPTIONAL: ≥ 50%
- ✅ EXCELLENT: ≥ 30%
- 🎯 VERY GOOD: ≥ 20%
- ⚠️ GOOD: ≥ 10%
- 📈 FAIR: ≥ 5%
- ❌ NEEDS IMPROVEMENT: < 5%

### Конфигурация оценки

| Параметр | Значение | Назначение |
|----------|----------|------------|
| temperature | 0.3 | Низкая для точных математических ответов |
| n_sampling | 8 | Множественная генерация для Pass@k |
| max_tokens | 32768 | Длинные рассуждения для сложных задач |
| top_p | 0.95 | Nucleus sampling |
| seed | 0 | Воспроизводимость результатов |

### Сравнение моделей

**Функция:** `compare_aime_results`

```
INPUT: list of results from multiple models
  ↓
Сравнительная таблица:
  - Accuracy по моделям
  - Pass@K по моделям
  - Improvement analysis (vs base model)
  - Performance breakdown по датасетам
  ↓
OUTPUT: aime_model_comparison.json
  - Лучшая модель
  - Детальное сравнение
```

---

