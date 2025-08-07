"""
API client utilities for making HTTP requests with retries and rate limiting.

This module provides a robust HTTP client for making API requests with built-in
retries, rate limiting, and error handling.
"""
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from dataclasses import dataclass
from functools import wraps

import aiohttp
import backoff
from aiohttp import ClientResponse, ClientSession, ClientTimeout
from pydantic import BaseModel, ValidationError

from ..config import settings
from .cache_utils import cache_function

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T', bound=BaseModel)
JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# Default timeout for API requests
DEFAULT_TIMEOUT = ClientTimeout(total=30.0)

@dataclass
class APIResponse:
    """Container for API response data."""
    status: int
    data: JsonType
    headers: Dict[str, str]
    
    def json(self) -> JsonType:
        """Get the response data as JSON."""
        return self.data
    
    @property
    def ok(self) -> bool:
        """Check if the request was successful (status code 2xx)."""
        return 200 <= self.status < 300
    
    def raise_for_status(self) -> None:
        """Raise an exception if the request was not successful."""
        if not self.ok:
            raise aiohttp.ClientResponseError(
                status=self.status,
                message=f"Request failed with status {self.status}",
                headers=self.headers
            )
    
    def model(self, model: Type[T]) -> T:
        """Parse the response data as a Pydantic model."""
        try:
            if isinstance(self.data, dict):
                return model.model_validate(self.data)
            raise ValueError("Response data is not a dictionary")
        except ValidationError as e:
            logger.error(f"Failed to validate response as {model.__name__}: {e}")
            raise

class APIClient:
    """A robust HTTP client for making API requests with retries and rate limiting."""
    
    def __init__(
        self,
        base_url: str = "",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the API client.
        
        Args:
            base_url: Base URL for all requests.
            api_key: API key for authentication.
            timeout: Default timeout in seconds for requests.
            max_retries: Maximum number of retries for failed requests.
            rate_limit: Maximum number of requests per second.
            headers: Default headers to include in all requests.
        """
        self.base_url = base_url.rstrip('/') if base_url else ""
        self.timeout = ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self._last_request_time: float = 0
        self._session: Optional[ClientSession] = None
        
        # Set up default headers
        self.headers = headers or {}
        if api_key:
            self.headers['Authorization'] = f"Bearer {api_key}"
        
        # Set default user agent
        self.headers.setdefault('User-Agent', 'PowerBI-Agent/1.0')
        self.headers.setdefault('Content-Type', 'application/json')
        self.headers.setdefault('Accept', 'application/json')
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def start(self) -> None:
        """Start the client session."""
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self.headers,
                raise_for_status=False,
            )
    
    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _ensure_session(self) -> ClientSession:
        """Ensure the client session is started."""
        if self._session is None or self._session.closed:
            await self.start()
        return self._session  # type: ignore
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting if configured."""
        if self.rate_limit and self.rate_limit > 0:
            min_interval = 1.0 / self.rate_limit
            now = time.monotonic()
            elapsed = now - self._last_request_time
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                await asyncio.sleep(sleep_time)
            
            self._last_request_time = time.monotonic()
    
    @backoff.on_exception(
        backoff.expo,
        (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionError,
            aiohttp.ClientResponseError,
        ),
        max_tries=3,
        jitter=backoff.full_jitter,
    )
    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> APIResponse:
        """Make an HTTP request with retries and rate limiting."""
        await self._rate_limit()
        session = await self._ensure_session()
        
        # Ensure JSON data is properly serialized
        if 'json' in kwargs and kwargs['json'] is not None:
            if 'headers' not in kwargs:
                kwargs['headers'] = {}
            kwargs['headers'].setdefault('Content-Type', 'application/json')
            kwargs['data'] = json.dumps(kwargs.pop('json'))
        
        try:
            async with session.request(method, url, **kwargs) as response:
                # Try to parse JSON response, fall back to text if not JSON
                try:
                    data = await response.json() if response.content_type == 'application/json' else await response.text()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    data = await response.text()
                
                # Log the request and response
                logger.debug(
                    "%s %s %s - Response: %s",
                    method.upper(),
                    url,
                    kwargs.get('json', {}),
                    response.status,
                )
                
                return APIResponse(
                    status=response.status,
                    data=data,
                    headers=dict(response.headers),
                )
        except Exception as e:
            logger.error("Request failed: %s %s - %s", method.upper(), url, str(e))
            raise
    
    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Send a GET request."""
        return await self._request(
            'GET',
            path,
            params=params,
            headers=headers,
            **kwargs,
        )
    
    async def post(
        self,
        path: str,
        json: Optional[JsonType] = None,
        data: Any = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Send a POST request."""
        return await self._request(
            'POST',
            path,
            json=json,
            data=data,
            headers=headers,
            **kwargs,
        )
    
    async def put(
        self,
        path: str,
        json: Optional[JsonType] = None,
        data: Any = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Send a PUT request."""
        return await self._request(
            'PUT',
            path,
            json=json,
            data=data,
            headers=headers,
            **kwargs,
        )
    
    async def delete(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Send a DELETE request."""
        return await self._request(
            'DELETE',
            path,
            headers=headers,
            **kwargs,
        )
    
    async def patch(
        self,
        path: str,
        json: Optional[JsonType] = None,
        data: Any = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Send a PATCH request."""
        return await self._request(
            'PATCH',
            path,
            json=json,
            data=data,
            headers=headers,
            **kwargs,
        )

# Global client instance for convenience
_global_client: Optional[APIClient] = None

def get_global_client() -> APIClient:
    """Get or create a global API client instance."""
    global _global_client
    if _global_client is None:
        _global_client = APIClient(
            api_key=settings.OPENAI_API_KEY if hasattr(settings, 'OPENAI_API_KEY') else None,
            timeout=30.0,
            max_retries=3,
            rate_limit=5.0,  # 5 requests per second by default
        )
    return _global_client

async def close_global_client() -> None:
    """Close the global API client if it exists."""
    global _global_client
    if _global_client is not None:
        await _global_client.close()
        _global_client = None

# Example usage:
# async with APIClient(base_url="https://api.example.com") as client:
#     response = await client.get("/some/endpoint")
#     if response.ok:
#         data = response.json()
#         print(data)
