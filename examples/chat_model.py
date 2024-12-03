import asyncio
from ai_processor import ChatProcessor

async def main():
    # Initialize the ChatProcessor
    processor = ChatProcessor(
        connection={
            "endpoint": "http://localhost:1234/v1/chat/completions",  # Used LMStudio
            "api_key": "dummy_api_key"
        },
        model_settings={
            "model_name": "meta-llama-3.1-8b-instruct",
            "max_tokens": 2048,
            "response_ratio": 0.3
        },
        log_level="DEBUG"
    )

    # Define input
    input_json = {
        "context": """
        Customer: John Doe; Order: 12345; Delivery: 2023-11-30
        Customer: Jane Smith; Order: 12346; Delivery: 2023-12-01
        Customer: Bob Johnson; Order: 12347; Delivery: 2023-12-02
        Customer: Alice Brown; Order: 12348; Delivery: 2023-12-03
        Customer: Charlie White; Order: 12349; Delivery: 2023-12-04
        """ * 50,  # Repeated text to increase volume
        "prompts": {
            "initial": "Summarize the following dataset in bullet points:",
            "follow_up_template": "Continue summarizing from: {last_chunk_end}"
        },
        "options": {
            "include_last_chunk": True,  # Include for sequential context
            "last_chunk_token_count": 50  # Token limit for context connection
        }
    }

    # Process the input
    result = await processor.process(
        context=input_json["context"],
        prompts=input_json["prompts"],
        options=input_json["options"]
    )

    # Print the result
    for chunk in result["chunks"]:
        print(f"Chunk {chunk['index']}:")
        print(f"Input: {chunk['input_text']}")
        print(f"Response: {chunk['response_text']}")

# Run the main function
asyncio.run(main())
