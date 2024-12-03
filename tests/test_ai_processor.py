import pytest
from ai_processor.ai_processor import ChatProcessor, EmbeddingsProcessor
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp


@pytest.fixture
def chat_processor():
    """
    Initialize a ChatProcessor instance for testing.

    Returns:
        ChatProcessor: Instance of ChatProcessor with mock settings.
    """
    connection = {
        "endpoint": "http://mock.endpoint",
        "api_key": "mock_key"
    }
    model_settings = {
        "model_name": "mock_chat_model",
        "max_tokens": 200,
        "response_ratio": 0.3
    }
    return ChatProcessor(connection, model_settings, log_level="DEBUG")


@pytest.fixture
def embeddings_processor():
    """
    Initialize an EmbeddingsProcessor instance for testing.

    Returns:
        EmbeddingsProcessor: Instance of EmbeddingsProcessor with mock settings.
    """
    connection = {
        "endpoint": "http://mock.endpoint",
        "api_key": "mock_key"
    }
    model_settings = {
        "model_name": "mock_embeddings_model",
        "max_tokens": 200
    }
    return EmbeddingsProcessor(connection, model_settings, log_level="DEBUG")


def test_calculate_chunk_size_chat(chat_processor):
    """
    Test chunk size calculation in ChatProcessor.

    Ensures that the calculated chunk size matches the expected value
    based on the provided response ratio and max tokens.
    """
    assert chat_processor._calculate_chunk_size() == 140


def test_split_into_chunks_chat(chat_processor):
    """
    Test splitting a context into chunks in ChatProcessor.

    Ensures that a valid context is split into multiple chunks correctly.
    """
    context = "Hello, how are you? This is a test. I am fine, thank you!"
    chunks = chat_processor._split_into_chunks(context)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_process_chat(chat_processor):
    """
    Test processing a context using ChatProcessor.

    Mocks the API response and verifies that the context is processed into chunks
    and responses are returned correctly.
    """
    context = "Hello, how are you? This is a test. I am fine, thank you!"
    prompts = {
        "initial": "Initial prompt",
        "follow_up_template": "Follow-up prompt with {last_chunk_end}"
    }

    mock_response = MagicMock()
    mock_response.__aenter__.return_value.json = AsyncMock(return_value={
        "choices": [{"message": {"content": "Mock response"}}]
    })

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        response = await chat_processor.process(context, prompts=prompts)
        assert response["status"] == "success"
        assert len(response["chunks"]) > 0
        assert response["chunks"][0]["response_text"] == "Mock response"


def test_calculate_chunk_size_embeddings(embeddings_processor):
    """
    Test chunk size calculation in EmbeddingsProcessor.

    Ensures that the calculated chunk size matches the max tokens value
    for embeddings mode.
    """
    assert embeddings_processor._calculate_chunk_size() == 200


def test_split_into_chunks_embeddings(embeddings_processor):
    """
    Test splitting a context into chunks in EmbeddingsProcessor.

    Ensures that a valid context is split into multiple chunks correctly.
    """
    context = "Hello, how are you? This is a test. I am fine, thank you!"
    chunks = embeddings_processor._split_into_chunks(context)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_process_embeddings(embeddings_processor):
    """
    Test processing a list of contexts using EmbeddingsProcessor.

    Mocks the API response and verifies that embeddings are generated
    for each input context.
    """
    context = ["Hello, how are you?", "This is a test."]
    mock_response = MagicMock()
    mock_response.__aenter__.return_value.json = AsyncMock(return_value={
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    })

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        response = await embeddings_processor.process(context)
        assert response["status"] == "success"
        assert len(response["embeddings"]) == len(context)
        assert response["embeddings"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_count_tokens(chat_processor):
    """
    Test token counting in ChatProcessor.

    Verifies that the _count_tokens method accurately counts the number of tokens
    in different strings.
    """
    assert chat_processor._count_tokens("hello world") == 2
    assert chat_processor._count_tokens("") == 0
    assert chat_processor._count_tokens("This is a test") == 4


def test_calculate_chunk_size_invalid_ratio(chat_processor):
    """
    Test invalid response ratio handling in ChatProcessor.

    Ensures that a ValueError is raised if the response ratio is outside
    the valid range (0 < ratio < 1).
    """
    chat_processor.response_ratio = 1.5
    with pytest.raises(ValueError):
        chat_processor._calculate_chunk_size()


@pytest.mark.asyncio
async def test_process_with_small_context(chat_processor):
    """
    Test processing a small context in ChatProcessor.

    Mocks the API response and verifies that small contexts are processed correctly.
    """
    context = "Hi!"
    prompts = {
        "initial": "Start the chat:",
        "follow_up_template": "Continue with {last_chunk_end}"
    }

    mock_response = MagicMock()
    mock_response.__aenter__.return_value.json = AsyncMock(return_value={
        "choices": [{"message": {"content": "Mock response"}}]
    })

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        response = await chat_processor.process(context, prompts=prompts)
        assert response["status"] == "success"
        assert len(response["chunks"]) == 1


def test_split_into_chunks_empty_context(chat_processor):
    """
    Test splitting an empty context into chunks.

    Verifies that splitting an empty context returns an empty list.
    """
    context = ""
    chunks = chat_processor._split_into_chunks(context)
    assert chunks == []


def test_reserved_tokens_exceed_limit(chat_processor):
    """
    Test exceeding token limit in ChatProcessor.

    Ensures that a ValueError is raised when reserved tokens fall below zero.
    """
    prompt = "a" * 1000
    chunk = "b" * 1000
    with pytest.raises(ValueError, match="Not enough tokens available for the response."):
        chat_processor._calculate_reserved_tokens(prompt, chunk)


def test_extract_last_chunk_end(chat_processor):
    """
    Test extracting the last chunk's end from a response.

    Verifies that the correct portion of the response text is extracted as the
    last chunk's end based on token count.
    """
    response_text = "This is a test response. It has some content."
    result = chat_processor._extract_last_chunk_end(response_text, last_chunk_token_count=3)
    assert result == "some content."


@pytest.mark.asyncio
async def test_call_model_with_mock(chat_processor):
    """
    Test the _call_model method in ChatProcessor with mocked API response.

    Mocks an API response and verifies that the method correctly handles
    asynchronous API calls and returns the expected response.
    """
    mock_response = MagicMock()
    mock_response.__aenter__.return_value.json = AsyncMock(return_value={
        "choices": [{"message": {"content": "Mock response"}}]
    })

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        response = await chat_processor._call_model("Mock prompt", "Mock input")
        assert response == "Mock response"
