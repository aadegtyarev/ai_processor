
# AI Processor

AI Processor — это Python-библиотека для работы с локальными AI-моделями, включая обработку текстовых чатов и генерацию эмбеддингов.

## Назначение

- Поддержка локальных моделей для интерактивных текстовых чатов.
- Генерация векторных представлений текста (эмбеддингов).
- Простая настройка и интеграция.
- Модульная структура для расширяемости.

## Возможности

- **Обработка текстовых сообщений**: Поддержка сложных диалогов с использованием моделей чата.
- **Генерация эмбеддингов**: Создание векторных представлений текста для последующего анализа.
- **Работа с токенами и чанками**: Разбиение текста на части для удобной обработки.
- **Логирование**: Поддержка различных уровней логирования для отладки.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/aadegtyarev/ai_processor.git
   cd ai_processor
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Установите библиотеку:
   ```bash
   pip install .
   ```

## Использование

### Пример 1: Обработка чатов (минимальный пример)

```python
from ai_processor.ai_processor import ChatProcessor

# Инициализация процессора чатов
chat_processor = ChatProcessor(
    connection={"endpoint": "http://localhost:1234/v1/chat/completions"},
    model_settings={
        "model_name": "chat-gpt-like-model",
        "max_tokens": 1024
    }
)

# Отправка сообщения в модель
response = chat_processor.process_message("Привет, как дела?")
print(response)
```

### Пример 2: Генерация эмбеддингов (минимальный пример)

```python
from ai_processor.ai_processor import EmbeddingsProcessor

# Инициализация процессора эмбеддингов
embeddings_processor = EmbeddingsProcessor(
    connection={"endpoint": "http://localhost:1234/v1/embeddings"},
    model_settings={
        "model_name": "embeddings-model",
        "max_tokens": 512
    }
)

# Генерация эмбеддингов
text = "Это пример текста."
embeddings = embeddings_processor.generate_embeddings(text)
print(embeddings)
```

## Формат вызовов API

### ChatProcessor

**Описание**: Используется для обработки текстовых сообщений с помощью локальной или удалённой модели.

- **`connection`** (обязательный параметр):
  - `endpoint` (строка): URL API или путь к локальной модели.
  - `api_key` (строка, опционально): Ключ API для удалённых сервисов.
  
- **`model_settings`** (обязательный параметр):
  - `model_name` (строка): Название используемой модели.
  - `max_tokens` (целое число): Максимальное количество токенов в ответе.
  - `response_ratio` (дробное число, опционально): Соотношение длины ответа к длине запроса.

**Пример полного запроса**:
```json
{
  "connection": {
    "endpoint": "http://localhost:1234/v1/chat/completions",
    "api_key": "your_api_key"
  },
  "model_settings": {
    "model_name": "chat-gpt-like-model",
    "max_tokens": 1024,
    "response_ratio": 0.3
  }
}
```

### EmbeddingsProcessor

**Описание**: Используется для генерации эмбеддингов текста.

- **`connection`** (обязательный параметр):
  - `endpoint` (строка): URL API или путь к локальной модели.
  - `api_key` (строка, опционально): Ключ API для удалённых сервисов.
  
- **`model_settings`** (обязательный параметр):
  - `model_name` (строка): Название используемой модели.
  - `max_tokens` (целое число, опционально): Максимальное количество токенов для разбиения текста на чанки.

**Пример полного запроса**:
```json
{
  "connection": {
    "endpoint": "http://localhost:1234/v1/embeddings",
    "api_key": "your_api_key"
  },
  "model_settings": {
    "model_name": "embeddings-model",
    "max_tokens": 512
  }
}
```

## Тестирование

Для запуска тестов выполните команду:
```bash
pytest tests/
```

## Участие в разработке

Мы приветствуем ваш вклад! Вы можете отправлять нам замечания или пул-реквесты через GitHub.

## Лицензия

Проект распространяется под лицензией MIT.
