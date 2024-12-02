import logging
import re
import aiohttp
from typing import Dict, Union, List

class BaseAIProcessor:
    def __init__(self, connection: Dict, model_settings: Dict, log_level: str = "INFO", logger: logging.Logger = None):
        """
        Initialize the BaseAIProcessor with necessary settings.

        Args:
            connection (Dict): A dictionary containing API endpoint and API key.
            model_settings (Dict): A dictionary containing model name and maximum tokens.
            log_level (str): The level of logging (e.g., 'DEBUG', 'INFO', 'WARNING').
            logger (logging.Logger): An existing logger object. If not provided, a new one will be created.

        Raises:
            ValueError: If any of the required settings are missing or invalid.
        """
        self.endpoint = connection["endpoint"]
        self.api_key = connection["api_key"]
        self.model_name = model_settings["model_name"]
        self.max_tokens = model_settings["max_tokens"]
        self.response_ratio = model_settings.get("response_ratio", None)  # Default: None

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=log_level)
        self.logger.setLevel(log_level)

        self.chunk_size = self._calculate_chunk_size()
        self.logger.info(f"{self.__class__.__name__} initialized with model={self.model_name} and max_tokens={self.max_tokens}.")

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given string.

        Args:
            text (str): The input string to count tokens from.

        Returns:
            int: The total number of tokens.
        """
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        return sum(max(1, len(token) // 5) for token in tokens)

    def _calculate_chunk_size(self) -> int:
        """
        Calculate the effective chunk size based on max_tokens and optional response_ratio.

        Returns:
            int: The calculated chunk size.
        """
        if self.response_ratio is None:  # Embeddings mode
            return self.max_tokens

        if not (0 < self.response_ratio < 1):  # Chat mode
            raise ValueError(f"response_ratio must be between 0 and 1. Got: {self.response_ratio}")

        return int(self.max_tokens * (1 - self.response_ratio))

    def _calculate_reserved_tokens(self, prompt: str, chunk: str, last_chunk_token_count: int = 0) -> int:
        """
        Calculate the number of tokens reserved for the response.

        Args:
            prompt (str): The initial or follow-up prompt.
            chunk (str): The input text to be processed in this chunk.
            last_chunk_token_count (int): The token count from the previous chunk.

        Returns:
            int: The number of tokens reserved for the response.
        """
        prompt_tokens = self._count_tokens(prompt)
        chunk_tokens = self._count_tokens(chunk)
        total_prompt_tokens = prompt_tokens + chunk_tokens + last_chunk_token_count
        reserved_tokens = self.max_tokens - total_prompt_tokens

        if reserved_tokens <= 0:
            raise ValueError("Not enough tokens available for the response.")

        return reserved_tokens

    def _extract_last_chunk_end(self, response_text: str, last_chunk_token_count: int = 50) -> str:
        """
        Extract the end of the last chunk from the response text.

        Args:
            response_text (str): The full response text.
            last_chunk_token_count (int): The token count of the previous chunk.

        Returns:
            str: The extracted end of the last chunk.
        """
        words = response_text.split()
        result, token_count = [], 0

        for word in reversed(words):
            word_tokens = self._count_tokens(word)
            if token_count + word_tokens > last_chunk_token_count:
                break
            result.append(word)
            token_count += word_tokens

        return " ".join(reversed(result))

    def _log_chunk_details(self, chunks: List[str]) -> None:
        """
        Log details about the chunks, including their total and individual token counts.

        Args:
            chunks (List[str]): A list of chunk texts.
        """
        total_tokens = sum(self._count_tokens(chunk) for chunk in chunks)
        self.logger.info(f"Context split into {len(chunks)} chunks with a total of {total_tokens} tokens.")
        for i, chunk in enumerate(chunks[:5]):
            self.logger.debug({"chunk_index": i + 1, "token_count": self._count_tokens(chunk)})
        if len(chunks) > 5:
            self.logger.debug(f"... {len(chunks) - 5} more chunks omitted.")

    def _format_with_truncation(self, text: str, max_length: int) -> str:
        """
        Format the text with truncation information.

        Args:
            text (str): The input text to format.
            max_length (int): The maximum allowed length of the truncated text.

        Returns:
            str: The formatted text with truncation info, if applicable.
        """
        if len(text) > max_length:
            truncated_part = len(text) - max_length
            return f"{text[:max_length]}...(+{truncated_part} chars)"
        return text

    def _split_into_chunks(self, context: str, last_chunk_end: str = "", include_last_chunk: bool = True) -> List[str]:
        """
        Split the context into chunks based on token limits.

        Args:
            context (str): The original text to split.
            last_chunk_end (str): The end of the previous chunk used for appending the current chunk.
            include_last_chunk (bool): Whether to include the last chunk in the result.

        Returns:
            List[str]: A list of chunk texts.
        """
        effective_chunk_size = self.chunk_size
        if include_last_chunk:
            effective_chunk_size -= self._count_tokens(last_chunk_end)

        if effective_chunk_size <= 0:
            raise ValueError("Effective chunk size is too small to process further.")

        chunks, current_chunk, current_token_count = [], [], 0

        for line in context.splitlines():
            line_token_count = self._count_tokens(line)

            if line_token_count > effective_chunk_size:
                words = line.split()
                for word in words:
                    word_token_count = self._count_tokens(word)
                    if current_token_count + word_token_count > effective_chunk_size:
                        chunks.append(' '.join(current_chunk))
                        current_chunk, current_token_count = [word], word_token_count
                    else:
                        current_chunk.append(word)
                        current_token_count += word_token_count
            elif current_token_count + line_token_count > effective_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_token_count = [line], line_token_count
            else:
                current_chunk.append(line)
                current_token_count += line_token_count

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        self._log_chunk_details(chunks)
        return chunks

    async def _call_model(self, *args, **kwargs) -> Union[str, List[float]]:
        """
        Abstract method to call the AI model. Subclasses must implement this method.

        Returns:
            Union[str, List[float]]: The response from the AI model or embeddings.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class EmbeddingsProcessor(BaseAIProcessor):
    async def _call_model(self, prompt: str, input_text: str) -> List[float]:
        """
        Call the embeddings model to generate an embedding vector for the given input text.

        Args:
            prompt (str): The initial prompt.
            input_text (str): The text to process for generating the embedding.

        Returns:
            List[float]: A list of floats representing the embedding.
        """
        payload = {"model": self.model_name, "input": input_text}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self.logger.debug(f"Calling embeddings model with input: {input_text[:50]}")

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload, headers=headers) as response:
                response_data = await response.json()
                embedding = response_data.get("data", [{}])[0].get("embedding", [])
                self.logger.debug(f"Received embedding of length {len(embedding)}")
                return embedding

    async def process(self, context: List[str], *args, **kwargs) -> Dict:
        """
        Process the given context to generate embeddings for each message.

        Args:
            context (List[str]): A list of messages to be processed.
            args: Additional arguments passed to the superclass method.
            kwargs: Additional keyword arguments passed to the superclass method.

        Returns:
            Dict: A dictionary containing the status and a list of dictionaries with
                 information about each message including its index, message text, and embedding.
        """
        if not isinstance(context, list):
            raise ValueError("Context must be a list of messages for embeddings mode.")

        embeddings = []
        for index, message in enumerate(context):
            self.logger.info(f"Processing message {index + 1}/{len(context)}")
            embedding = await self._call_model("", message)
            embeddings.append({"index": index, "message": message, "embedding": embedding})

        return {"status": "success", "embeddings": embeddings}

class ChatProcessor(BaseAIProcessor):
    async def _call_model(self, prompt: str, input_text: str) -> str:
        """
        Call the chat model to generate a response based on the given prompt and input text.

        Args:
            prompt (str): The initial prompt.
            input_text (str): The user's input for generating the response.

        Returns:
            str: A string representing the generated response.
        """
        self.logger.debug(
            f"Calling chat model with: "
            f"prompt={self._format_with_truncation(prompt, 50)}, "
            f"chunk={self._format_with_truncation(input_text, 50)}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ],
            "max_tokens": self.chunk_size,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload, headers=headers) as response:
                response_data = await response.json()
                response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                self.logger.debug(f"Received response: {self._format_with_truncation(response_text, 50)}")
                return response_text

    async def process(self, context: str, prompts: Dict, options: Dict = None) -> Dict:
        """
        Process the given context to generate responses based on the provided prompts and options.

        Args:
            context (str): The original text to split into chunks.
            prompts (Dict): A dictionary containing initial and follow-up prompt templates.
            options (Dict): Optional configuration for processing, including whether to include the last chunk.

        Returns:
            Dict: A dictionary containing the status and a list of dictionaries with
                 information about each input text (chunk) including its index, response text, and original input.
        """
        include_last_chunk = options.get("include_last_chunk", False) if options else False
        last_chunk_end = ""  # Initial last chunk is empty
        last_chunk_token_count = 0

        chunks = self._split_into_chunks(context, last_chunk_end, include_last_chunk)
        results = []

        for index, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {index + 1}/{len(chunks)}")

            # Use initial or follow-up prompt
            prompt = prompts["initial"] if index == 0 else prompts["follow_up_template"].format(last_chunk_end=last_chunk_end)

            # Calculate token counts
            prompt_tokens = self._count_tokens(prompt)
            chunk_tokens = self._count_tokens(chunk)
            total_tokens = prompt_tokens + chunk_tokens + last_chunk_token_count
            reserved_tokens = self._calculate_reserved_tokens(prompt, chunk, last_chunk_token_count)

            # Log detailed token information
            self.logger.debug(
                f"Token details for chunk {index + 1}: "
                f"chunk_tokens={chunk_tokens}, prompt_tokens={prompt_tokens}, "
                f"total_request_tokens={total_tokens}, reserved_tokens={reserved_tokens}"
            )

            # Log truncated prompt and chunk preview
            self.logger.debug(
                f"Prompt preview: {self._format_with_truncation(prompt, 50)}"
            )
            self.logger.debug(
                f"Chunk preview: {self._format_with_truncation(chunk, 50)}"
            )

            # Call model and get response
            response_text = await self._call_model(prompt, chunk)
            results.append({"index": index + 1, "input_text": chunk, "response_text": response_text})

            if include_last_chunk:
                last_chunk_end = self._extract_last_chunk_end(response_text)
                last_chunk_token_count = self._count_tokens(last_chunk_end)

        return {"status": "success", "chunks": results}
