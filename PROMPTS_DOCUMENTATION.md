# AI Prompts Documentation

Этот документ содержит все промпты для AI, используемые в приложении Unsloth, сгруппированные по тематикам.

## Оглавление

1. [Промпты для генерации синтетических данных](#промпты-для-генерации-синтетических-данных)
2. [Промпты для математических задач](#промпты-для-математических-задач)
3. [Промпты для обучающих датасетов](#промпты-для-обучающих-датасетов)
4. [Промпты для Vision моделей](#промпты-для-vision-моделей)

---

## Промпты для генерации синтетических данных

### 1. Summary Generation Prompt

**Расположение:** `unsloth/dataprep/synthetic_configs.py:73-74`

**Назначение:** Используется для генерации краткого содержания документа. Промпт запрашивает создание резюме в 3-5 предложениях, фокусируясь на основной теме и ключевых концепциях.

**Промпт:**
```
Summarize this document in 3-5 sentences, focusing on the main topic and key concepts.
```

**Параметры конфигурации:**
- Используется с моделью, указанной в vllm.model
- Temperature и top_p настраиваются в секции generation конфигурации

---

### 2. QA Pair Generation Prompt

**Расположение:** `unsloth/dataprep/synthetic_configs.py:77-97`

**Назначение:** Создает пары вопрос-ответ из текста для обучения LLM моделей. Промпт гарантирует, что вопросы основаны на важных фактах из текста, а ответы непосредственно поддерживаются текстом. Выходной формат - JSON.

**Промпт:**
```
Create {num_pairs} question-answer pairs from this text for LLM training.

Rules:
1. Questions must be about important facts in the text
2. Answers must be directly supported by the text
3. Return JSON format only:

[
  {{
    "question": "Question 1?",
    "answer": "Answer 1."
  }},
  {{
    "question": "Question 2?",
    "answer": "Answer 2."
  }}
]

Text:
{text}
```

**Параметры:**
- `{num_pairs}` - количество пар вопрос-ответ для генерации (по умолчанию задается в generation.num_pairs)
- `{text}` - текст для обработки

**Конфигурация:**
- Temperature: задается в generation.temperature
- Top_p: задается в generation.top_p
- Max tokens: задается в generation.max_tokens

---

### 3. QA Rating Prompt

**Расположение:** `unsloth/dataprep/synthetic_configs.py:100-111`

**Назначение:** Оценивает качество сгенерированных пар вопрос-ответ по шкале от 1 до 10. Используется для фильтрации низкокачественных QA пар. Возвращает JSON с оценками.

**Промпт:**
```
Rate each of these question-answer pairs for quality and return exactly this JSON format:

[
  {{"question": "same question text", "answer": "same answer text", "rating": n}}
]

Where n is a number from 1-10.

DO NOT include any text outside of the JSON array, just return valid JSON:

{pairs}
```

**Параметры:**
- `{pairs}` - список пар вопрос-ответ для оценки

**Конфигурация:**
- Temperature: задается в cleanup.temperature (по умолчанию 0.3 для более консистентных оценок)
- Batch size: задается в cleanup.batch_size
- Threshold: задается в cleanup.threshold (минимальная оценка для прохождения)

---

## Промпты для математических задач

### 4. AIME Mathematical Problem Solver System Prompt

**Расположение:** `tests/utils/aime_eval.py:116-118`

**Назначение:** System prompt для решения математических задач из датасета AIME (American Invitational Mathematics Examination). Инструктирует модель решать задачи пошагово и четко предоставлять финальный ответ.

**Промпт:**
```
You are a mathematical problem solver. Solve the given problem step by step and provide your final answer clearly.
```

**Использование:**
- Используется как system message в массиве промптов
- Применяется для задач из combined AIME dataset (test2024 + test2025-I + test2025-II)
- Ожидается численный ответ в диапазоне 0-999

---

### 5. AIME Problem User Prompt Template

**Расположение:** `tests/utils/aime_eval.py:121`

**Назначение:** User prompt для представления математической задачи модели. Запрашивает пошаговое решение и финальный численный ответ.

**Промпт:**
```
Problem: {problem}

Solve this step by step and provide your final numerical answer.
```

**Параметры:**
- `{problem}` - текст математической задачи из датасета

**Конфигурация оценки:**
- Temperature: 0.3 (по умолчанию)
- n_sampling: 8 (количество попыток на задачу)
- max_tokens: 32768
- top_p: 0.95
- Используется Pass@k метрика для оценки

**Формат ответа:**
- Извлекаются числа от 0 до 999
- Поддерживаются паттерны: "The answer is X", "\\boxed{X}", standalone numbers

---

## Промпты для обучающих датасетов

### 6. GSM8K Dataset System Prompt

**Расположение:** `tests/saving/language_models/test_save_merged_grpo_model.py:112-116`

**Назначение:** System prompt для форматирования датасета GSM8K для обучения модели. Инструктирует модель размещать процесс мышления между специальными тегами `<reasoning>` и предоставлять финальное численное решение между тегами `<answer>`.

**Промпт:**
```
You are given a problem. Think about the problem and reason step by step. Place your thinking process between <reasoning> and </reasoning>. Then, provide your final numerical solution between <answer></answer>
```

**Использование:**
- Применяется в функции `prepare_gsm8k_dataset`
- Используется для форматирования задач GSM8K
- Работает с GRPO (Group Relative Policy Optimization) обучением

**Формат данных:**
```python
{
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ],
    "answer": extracted_answer
}
```

**Теги форматирования:**
- `<reasoning>...</reasoning>` - процесс размышления
- `<answer>...</answer>` - финальный численный ответ

---

### 7. LIMO Dataset System Prompt

**Расположение:** `tests/saving/language_models/test_save_merged_grpo_model.py:134-141`

**Назначение:** System prompt для форматирования датасета LIMO для SFT (Supervised Fine-Tuning) обучения. Более подробный промпт, который объясняет ассистенту формат ответа с примерами тегов.

**Промпт:**
```
You are a helpful reasoning assistant. When given a problem, think through it step by step and provide your answer in the following format:

<reasoning>
[Your detailed step-by-step reasoning and solution process]
</reasoning>
<answer>
[Your final numerical answer]
</answer>
```

**Использование:**
- Применяется в функции `prepare_limo_dataset`
- Используется для SFT training
- Создает более структурированный формат с разделением reasoning и answer

**Формат данных:**
```python
{
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": formatted_response}
    ]
}
```

**Assistant response format:**
```
<reasoning>
{solution}
</reasoning>
<answer>
{answer}
</answer>
```

---

## Промпты для Vision моделей

### 8. French OCR System Prompt

**Расположение:** `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py:62`

**Назначение:** System prompt для Vision модели, которая выполняет OCR (Optical Character Recognition) на французском языке. Определяет роль модели как эксперта в распознавании французского текста.

**Промпт:**
```
You are an expert french ocr system.
```

**Использование:**
- Применяется для обучения Vision моделей на датасете OCR
- Используется с Qwen2-VL-7B-Instruct модель
- Работает с датасетом "lbourdois/OCR-liboaccn-OPUS-MIT-5M-clean"

**Формат данных:**
```python
{
    "messages": [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}]
        }
    ]
}
```

**Метрики оценки:**
- WER (Word Error Rate) - частота ошибок на уровне слов
- CER (Character Error Rate) - частота ошибок на уровне символов

**Конфигурация обучения:**
- Model: unsloth/Qwen2-VL-7B-Instruct
- Max seq length: 2048
- LoRA rank (r): 16
- LoRA alpha: 32
- Training samples: 2000
- Evaluation samples: 200

---

## Общая информация

### Категории промптов

Все промпты в приложении Unsloth можно разделить на следующие категории:

1. **Генерация синтетических данных** (3 промпта)
   - Summary generation
   - QA pair generation
   - QA rating

2. **Математические задачи** (2 промпта)
   - AIME problem solver
   - Problem presentation template

3. **Обучающие датасеты** (2 промпта)
   - GSM8K dataset formatting
   - LIMO dataset formatting

4. **Vision модели** (1 промпт)
   - French OCR system

### Общие паттерны

#### Структура промптов

Большинство промптов следуют паттерну трех ролей:
- **system** - определяет роль и поведение модели
- **user** - содержит запрос или задачу
- **assistant** - содержит ответ (используется при обучении)

#### Теги форматирования

В приложении используются следующие специальные теги для структурирования ответов:

- `<reasoning>...</reasoning>` - процесс размышления модели
- `<answer>...</answer>` - финальный ответ
- `\boxed{...}` - математическая нотация для ответа (AIME)

#### Конфигурационные параметры

Основные параметры, используемые при генерации:
- **temperature** - контролирует креативность (0.3-1.5)
- **top_p** - nucleus sampling (обычно 0.95)
- **max_tokens** - максимальная длина ответа (512-32768)
- **n_sampling** - количество попыток генерации

### Файловая структура

Промпты расположены в следующих файлах:
- `unsloth/dataprep/synthetic_configs.py` - синтетические данные
- `tests/utils/aime_eval.py` - математические задачи
- `tests/saving/language_models/test_save_merged_grpo_model.py` - обучающие датасеты
- `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py` - Vision модели

---

**Дата создания:** 2025-11-13
**Всего промптов:** 8

