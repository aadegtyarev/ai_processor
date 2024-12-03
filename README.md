
# AI Processor

AI Processor is a Python library for working with local AI models, including processing text-based chats and generating embeddings.

## Purpose

- Support for local models for interactive text-based chats.
- Generation of vector representations of text (embeddings).
- Simple configuration and integration.
- Modular structure for extensibility.

## Features

- **Text Message Processing**: Support for complex dialogues using chat models.
- **Embedding Generation**: Creation of vector representations of text for further analysis.
- **Token and Chunk Handling**: Splitting long texts into chunks for easy processing.
- **Logging**: Support for different logging levels for debugging.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aadegtyarev/ai_processor.git
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

## Usage

### Example 1: Chat Processing (minimal example)

```python
from ai_processor.ai_processor import ChatProcessor

# Initialize the chat processor
chat_processor = ChatProcessor(
    connection={"endpoint": "http://localhost:1234/v1/chat/completions"},
    model_settings={
        "model_name": "chat-gpt-like-model",
        "max_tokens": 1024
    }
)

# Send a message to the model
response = chat_processor.process_message("Hello, how are you?")
print(response)
```

### Example 2: Embedding Generation (minimal example)

```python
from ai_processor.ai_processor import EmbeddingsProcessor

# Initialize the embeddings processor
embeddings_processor = EmbeddingsProcessor(
    connection={"endpoint": "http://localhost:1234/v1/embeddings"},
    model_settings={
        "model_name": "embeddings-model",
        "max_tokens": 512
    }
)

# Generate embeddings
text = "This is a sample text."
embeddings = embeddings_processor.generate_embeddings(text)
print(embeddings)
```

## API Call Format

### ChatProcessor

**Description**: Used for processing text messages using either local or remote models.

- **`connection`** (required parameter):
  - `endpoint` (string): API URL or path to the local model.
  - `api_key` (string, optional): API key for remote services.
  
- **`model_settings`** (required parameter):
  - `model_name` (string): Name of the model being used.
  - `max_tokens` (integer): Maximum number of tokens in the response.
  - `response_ratio` (float, optional): Ratio of the response length to the input query length.

**Example full request**:
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

**Description**: Used for generating text embeddings.

- **`connection`** (required parameter):
  - `endpoint` (string): API URL or path to the local model.
  - `api_key` (string, optional): API key for remote services.
  
- **`model_settings`** (required parameter):
  - `model_name` (string): Name of the model being used.
  - `max_tokens` (integer, optional): Maximum number of tokens for splitting the text into chunks.

**Example full request**:
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

## Testing

To run tests, use the following command:
```bash
pytest tests/
```

## Contributing

We welcome your contributions! You can submit issues or pull requests via GitHub.

## License

This project is licensed under the MIT License.
