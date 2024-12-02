import asyncio
from ai_processor import EmbeddingsProcessor

async def main():
    # Initialize the EmbeddingsProcessor
    processor = EmbeddingsProcessor(
        connection={
            "endpoint": "http://localhost:1234/v1/embeddings",  # Embeddings endpoint
            "api_key": "dummy_api_key"
        },
        model_settings={
            "model_name": "text-embedding-nomic-embed-text-v1.5",
            "max_tokens": 1024
        },
        log_level="DEBUG"
    )

    # Define input for embeddings
    embedding_messages = [
        "This is the first test message.",
        "Here comes another example of embedding generation.",
        "We are testing the embedding functionality with multiple inputs."
    ]

    # Process the input for embeddings
    embedding_result = await processor.process(context=embedding_messages)

    # Print the result for embeddings
    if embedding_result["status"] == "success":
        print("\nEmbeddings generated successfully:")
        for embedding_data in embedding_result["embeddings"]:
            print(f"Message {embedding_data['index']}: {embedding_data['message']}")
            print(f"Embedding: {embedding_data['embedding'][:5]}...")  # Truncated for readability
    else:
        print("\nError in embedding processing:")
        print(embedding_result["message"])

# Run the main function
asyncio.run(main())
