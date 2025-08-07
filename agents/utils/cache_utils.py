"""
Caching utilities for the application.

This module provides a simple caching mechanism to store and retrieve
computed results, reducing redundant computations and database queries.
"""
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic function typing
T = TypeVar('T')

class Cache:
    """A simple in-memory cache with TTL (time-to-live) support."""
    
    def __init__(self):
        """Initialize the cache with an empty dictionary."""
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Any:
        """Get a value from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached value or None if not found or expired.
        """
        if key not in self._cache:
            return None
            
        cached = self._cache[key]
        
        # Check if the cached item has expired
        if 'expires' in cached and cached['expires'] < time.time():
            del self._cache[key]
            return None
            
        return cached.get('value')
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None
    ) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time to live in seconds. If None, the value won't expire.
        """
        cache_entry = {'value': value}
        
        if ttl is not None:
            cache_entry['expires'] = time.time() + ttl
            
        self._cache[key] = cache_entry
    
    def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: The cache key to delete.
        """
        if key in self._cache:
            del self._cache[key]
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()
    
    def get_or_set(
        self, 
        key: str, 
        default: Any = None, 
        ttl: Optional[float] = None
    ) -> Any:
        """Get a value from the cache, or set it if not present.
        
        Args:
            key: The cache key.
            default: The default value to set if the key is not in the cache.
            ttl: Time to live in seconds for the default value if set.
            
        Returns:
            The cached or default value.
        """
        value = self.get(key)
        if value is None and default is not None:
            self.set(key, default, ttl=ttl)
            return default
        return value

def generate_cache_key(
    func: Callable, 
    *args: Any, 
    **kwargs: Any
) -> str:
    """Generate a cache key for a function call.
    
    Args:
        func: The function being cached.
        *args: Positional arguments passed to the function.
        **kwargs: Keyword arguments passed to the function.
        
    Returns:
        A string representing a unique cache key.
    """
    # Create a string representation of the function and its arguments
    key_parts = [
        func.__module__ or '',
        func.__qualname__,
        json.dumps(args, sort_keys=True, default=str),
        json.dumps(kwargs, sort_keys=True, default=str)
    ]
    
    # Create a hash of the key parts
    key_string = ':'.join(str(part) for part in key_parts)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def cached(
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None,
    cache: Optional[Cache] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to cache the result of a function call.
    
    Args:
        ttl: Time to live in seconds for cached results.
        key_func: Optional function to generate cache keys.
        cache: The cache instance to use. If None, a new instance will be created.
        
    Returns:
        A decorator function.
    """
    if cache is None:
        cache = Cache()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """The actual decorator function."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate a cache key
            if key_func is not None:
                key = key_func(*args, **kwargs)
            else:
                key = generate_cache_key(func, *args, **kwargs)
            
            # Try to get the result from the cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__} with key {key}")
                return cached_result
            
            # If not in cache, compute the result
            logger.debug(f"Cache miss for {func.__name__} with key {key}")
            result = func(*args, **kwargs)
            
            # Store the result in the cache
            cache.set(key, result, ttl=ttl)
            
            return result
        
        # Add cache management methods to the wrapper
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        
        return wrapper
    
    return decorator

# Global cache instance
default_cache = Cache()

def get_cache() -> Cache:
    """Get the default cache instance."""
    return default_cache

def clear_cache() -> None:
    """Clear the default cache."""
    default_cache.clear()

def cache_function(
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Convenience decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds for cached results.
        key_func: Optional function to generate cache keys.
        
    Returns:
        A decorator function that uses the default cache.
    """
    return cached(ttl=ttl, key_func=key_func, cache=default_cache)

def memoize(
    func: Optional[Callable[..., T]] = None,
    ttl: Optional[float] = None
) -> Union[Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
    """A simple memoization decorator with TTL support.
    
    This is a simpler alternative to @cached for common use cases.
    
    Args:
        func: The function to memoize (for direct decoration).
        ttl: Time to live in seconds for cached results.
        
    Returns:
        A memoized version of the function.
    """
    if func is not None:
        # Direct decoration: @memoize
        return cached(ttl=ttl)(func)
    else:
        # Parameterized decoration: @memoize(ttl=60)
        return cached(ttl=ttl)
