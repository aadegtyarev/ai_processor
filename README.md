
# AI Processor

**AI Processor** is a Python library for working with local and remote AI models. It provides convenient tools for generating responses via chat models and obtaining text embeddings.

## Key Features

- Connect to local or remote model APIs.
- Process large texts by splitting them into chunks.
- Generate responses based on context and prompts.
- Generate embeddings for a list of texts.
- Asynchronous approach for high performance.
- Detailed logging for debugging.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_repository/ai_processor.git
   cd ai_processor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the library:
   ```bash
   pip install .
   ```

## Quick Start

### 1. Connection Configuration

To use the module, you need to specify the API connection settings:

```json
{
  "endpoint": "http://localhost:1234/v1/chat/completions",
  "api_key": "your_api_key"
}
```

### 2. Model Configuration

Model parameters define its behavior:

```json
{
  "model_name": "meta-llama-3.1-8b-instruct",
  "max_tokens": 2048,
  "response_ratio": 0.3
}
```

- **`model_name`**: The name of the model (required).
- **`max_tokens`**: Maximum number of tokens in the response (required).
- **`response_ratio`**: The proportion of tokens reserved for the response (optional, only for chat models).

### 3. Example: Using ChatProcessor

#### Request

```python
from ai_processor import ChatProcessor
import asyncio

async def main():
    processor = ChatProcessor(
        connection={
            "endpoint": "http://localhost:1234/v1/chat/completions",
            "api_key": "your_api_key"
        },
        model_settings={
            "model_name": "meta-llama-3.1-8b-instruct",
            "max_tokens": 2048,
            "response_ratio": 0.3
        }
    )

    context = "Example text for processing."
    prompts = {
        "initial": "Summarize briefly:",
        "follow_up_template": "Continue from: {last_chunk_end}"
    }
    options = {"include_last_chunk": True, "last_chunk_token_count": 50}

    response = await processor.process(context=context, prompts=prompts, options=options)
    print(response)

asyncio.run(main())
```

#### Example Response
```json
{
  "status": "success",
  "chunks": [
    {
      "index": 1,
      "input_text": "Example text for processing.",
      "response_text": "Brief summary: the text was successfully processed."
    }
  ]
}
```

### 4. Example: Using EmbeddingsProcessor

#### Request

```python
from ai_processor import EmbeddingsProcessor
import asyncio

async def main():
    processor = EmbeddingsProcessor(
        connection={
            "endpoint": "http://localhost:1234/v1/embeddings",
            "api_key": "your_api_key"
        },
        model_settings={
            "model_name": "text-embedding-nomic-embed-text-v1.5",
            "max_tokens": 1024
        }
    )

    messages = [
        "Example of the first text.",
        "Another text for embedding generation."
    ]

    response = await processor.process(context=messages)
    print(response)

asyncio.run(main())
```

#### Example Response
```json
{
  "status": "success",
  "embeddings": [
    {
      "index": 0,
      "message": "Example of the first text.",
      "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
    },
    {
      "index": 1,
      "message": "Another text for embedding generation.",
      "embedding": [0.6, 0.7, 0.8, 0.9, 1.0]
    }
  ]
}
```

### 5. Request and Response Formats

#### For ChatProcessor
**Request**:
```json
{
  "context": "Text for analysis.",
  "prompts": {
    "initial": "Write a brief summary:",
    "follow_up_template": "Continue from: {last_chunk_end}"
  },
  "options": {
    "include_last_chunk": true,
    "last_chunk_token_count": 50
  }
}
```

**Response**:
```json
{
  "status": "success",
  "chunks": [
    {
      "index": 1,
      "input_text": "Text for analysis.",
      "response_text": "Brief summary: the text was processed."
    }
  ]
}
```

#### For EmbeddingsProcessor
**Request**:
```json
{
  "context": [
    "First text for analysis.",
    "Second text for embedding generation."
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "embeddings": [
    {
      "index": 0,
      "message": "First text for analysis.",
      "embedding": [0.1, 0.2, 0.3, ...]
    },
    {
      "index": 1,
      "message": "Second text for embedding generation.",
      "embedding": [0.6, 0.7, 0.8, ...]
    }
  ]
}
```

### 6. Logging

For debugging, you can set the logging level:
- `DEBUG`: Detailed information.
- `INFO`: Basic events.
- `ERROR`: Errors only.

Example configuration:
```python
processor = ChatProcessor(
    connection={"endpoint": "http://localhost:1234/v1/chat/completions", "api_key": "your_api_key"},
    model_settings={"model_name": "meta-llama-3.1-8b-instruct", "max_tokens": 2048},
    log_level="DEBUG"
)
```

## Testing

To run tests, execute:
```bash
pytest tests/
```

## Contributing

We welcome your contributions! You can submit issues or pull requests via GitHub.

## License

This project is licensed under the MIT License.
