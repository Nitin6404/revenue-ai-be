"""
Utilities for interacting with the OpenAI API.

This module provides a high-level interface for working with OpenAI's API,
including text generation, chat completions, and embeddings.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Union, Literal, AsyncGenerator
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, validator

from ..config import settings
from .cache_utils import cache_function
from .api_client import APIClient

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
ModelName = Literal[
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-3.5-turbo",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
]

# Default model names
DEFAULT_CHAT_MODEL: ModelName = "gpt-3.5-turbo"
DEFAULT_EMBEDDING_MODEL: ModelName = "text-embedding-3-small"

# Initialize the OpenAI client
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

class Message(BaseModel):
    """A message in a chat conversation."""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        data = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.function_call:
            data["function_call"] = self.function_call
        return data

class FunctionCall(BaseModel):
    """A function call in a chat message."""
    name: str
    arguments: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the function call to a dictionary."""
        return {"name": self.name, "arguments": self.arguments}

class Function(BaseModel):
    """A function that can be called by the model."""
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the function to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

class ChatResponse(BaseModel):
    """A response from the chat completion API."""
    id: str
    object: str
    created: int
    model: str
    usage: Dict[str, int]
    choices: List[Dict[str, Any]]
    finish_reason: Optional[str] = None
    message: Optional[Message] = None
    function_call: Optional[FunctionCall] = None

    @property
    def content(self) -> str:
        """Get the content of the first choice."""
        if self.choices and len(self.choices) > 0:
            return self.choices[0].get("message", {}).get("content", "")
        return ""

    @property
    def function_arguments(self) -> Optional[Dict[str, Any]]:
        """Get the function call arguments as a dictionary."""
        if self.function_call:
            try:
                return json.loads(self.function_call.arguments)
            except json.JSONDecodeError:
                logger.warning("Failed to parse function arguments as JSON")
        return None

class EmbeddingResponse(BaseModel):
    """A response from the embeddings API."""
    object: str
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

    @property
    def embedding(self) -> List[float]:
        """Get the first embedding from the response."""
        if self.data and len(self.data) > 0:
            return self.data[0].get("embedding", [])
        return []

class OpenAIClient:
    """A client for interacting with the OpenAI API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize the OpenAI client.
        
        Args:
            api_key: Your OpenAI API key. If not provided, will use OPENAI_API_KEY from settings.
            organization: Your OpenAI organization ID.
            timeout: Timeout in seconds for API requests.
            max_retries: Maximum number of retries for failed requests.
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.organization = organization or getattr(settings, "OPENAI_ORGANIZATION", None)
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize the async client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
            timeout=timeout,
            max_retries=max_retries,
        )
    
    async def chat_completion(
        self,
        messages: List[Union[Dict[str, Any], Message]],
        model: ModelName = DEFAULT_CHAT_MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        functions: Optional[List[Union[Dict[str, Any], Function]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatResponse, AsyncGenerator[ChatResponse, None]]:
        """Generate a chat completion.
        
        Args:
            messages: List of message dictionaries or Message objects.
            model: The model to use for completion.
            temperature: Controls randomness. Higher values make the output more random.
            max_tokens: Maximum number of tokens to generate.
            functions: List of functions the model can call.
            function_call: Controls how the model responds to function calls.
            stream: Whether to stream the response.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            A ChatResponse object or an async generator of ChatResponse objects if streaming.
        """
        # Convert messages to dictionaries if they're Message objects
        message_dicts = [
            msg.to_dict() if isinstance(msg, Message) else msg
            for msg in messages
        ]
        
        # Convert functions to dictionaries if they're Function objects
        function_dicts = None
        if functions is not None:
            function_dicts = [
                func.to_dict() if isinstance(func, Function) else func
                for func in functions
            ]
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=message_dicts,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                functions=function_dicts,  # type: ignore
                function_call=function_call,
                stream=stream,
                **kwargs,
            )
            
            if stream:
                # For streaming responses, return an async generator
                async def stream_generator():
                    async for chunk in response:
                        yield ChatResponse(**chunk.model_dump())
                return stream_generator()
            else:
                # For non-streaming responses, return a single ChatResponse
                return ChatResponse(**response.model_dump())
                
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
    
    @cache_function(ttl=3600)  # Cache embeddings for 1 hour
    async def get_embedding(
        self,
        text: str,
        model: ModelName = DEFAULT_EMBEDDING_MODEL,
        **kwargs,
    ) -> List[float]:
        """Get an embedding for the given text.
        
        Args:
            text: The text to embed.
            model: The embedding model to use.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            A list of floats representing the embedding.
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=model,
                **kwargs,
            )
            
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            return []
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
    
    async def get_embeddings_batch(
        self,
        texts: List[str],
        model: ModelName = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 100,
        **kwargs,
    ) -> List[List[float]]:
        """Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed.
            model: The embedding model to use.
            batch_size: Number of texts to process in each batch.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            A list of embeddings, one for each input text.
        """
        embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = await self.client.embeddings.create(
                    input=batch,
                    model=model,
                    **kwargs,
                )
                
                # Extract embeddings from the response
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error in embedding batch {i//batch_size + 1}: {str(e)}")
                # Add empty lists for failed embeddings
                embeddings.extend([[] for _ in batch])
        
        return embeddings

# Global client instance for convenience
_global_client: Optional[OpenAIClient] = None

def get_openai_client() -> OpenAIClient:
    """Get or create a global OpenAI client instance."""
    global _global_client
    if _global_client is None:
        _global_client = OpenAIClient()
    return _global_client

# Example usage:
# client = get_openai_client()
# 
# # Chat completion
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
# ]
# response = await client.chat_completion(messages)
# print(response.content)
# 
# # Get embeddings
# embedding = await client.get_embedding("Hello, world!")
# print(f"Embedding length: {len(embedding)}")
# 
# # Batch embeddings
# texts = ["First text", "Second text"]
# embeddings = await client.get_embeddings_batch(texts)
# for text, emb in zip(texts, embeddings):
#     print(f"{text}: {len(emb)} dimensions")
