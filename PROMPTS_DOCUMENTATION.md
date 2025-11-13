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

