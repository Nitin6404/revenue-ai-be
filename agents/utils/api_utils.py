"""
Utility functions for handling API responses and errors.

This module provides helper functions for creating consistent API responses
and handling errors in a standardized way.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

class APIResponse(BaseModel):
    """Standard API response model."""
    status: str
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

class APIError(Exception):
    """Base exception for API errors."""
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        message: str = "An error occurred",
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.status_code = status_code
        self.message = message
        self.error_type = error_type or "api_error"
        self.details = details or {}
        self.extra = kwargs
        super().__init__(message)

class ValidationError(APIError):
    """Exception for validation errors."""
    def __init__(
        self,
        message: str = "Validation error",
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message=message,
            error_type="validation_error",
            details=details,
            **kwargs
        )

class NotFoundError(APIError):
    """Exception for not found errors."""
    def __init__(
        self,
        resource: str = "resource",
        id: Optional[Union[str, int]] = None,
        **kwargs
    ):
        message = f"{resource} not found"
        if id is not None:
            message = f"{resource} with id '{id}' not found"
            
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=message,
            error_type="not_found",
            details={"resource": resource, "id": id},
            **kwargs
        )

class UnauthorizedError(APIError):
    """Exception for unauthorized access errors."""
    def __init__(
        self,
        message: str = "Unauthorized",
        **kwargs
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message=message,
            error_type="unauthorized",
            **kwargs
        )

class ForbiddenError(APIError):
    """Exception for forbidden access errors."""
    def __init__(
        self,
        message: str = "Forbidden",
        **kwargs
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            message=message,
            error_type="forbidden",
            **kwargs
        )

def create_response(
    data: Optional[Any] = None,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK,
    meta: Optional[Dict[str, Any]] = None,
    **kwargs
) -> JSONResponse:
    """Create a standardized API response.
    
    Args:
        data: The response data.
        message: Optional message to include in the response.
        status_code: HTTP status code.
        meta: Optional metadata to include in the response.
        **kwargs: Additional fields to include in the response.
        
    Returns:
        A FastAPI JSONResponse with the formatted data.
    """
    response_data = {
        "status": "success" if status_code < 400 else "error",
    }
    
    if data is not None:
        response_data["data"] = data
        
    if message:
        response_data["message"] = message
        
    if meta:
        response_data["meta"] = meta
    
    # Add any additional fields
    response_data.update(kwargs)
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )

def create_error_response(
    message: str,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    error_type: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    **kwargs
) -> JSONResponse:
    """Create a standardized error response.
    
    Args:
        message: Error message.
        status_code: HTTP status code.
        error_type: Type of error (e.g., 'validation_error', 'not_found').
        details: Additional error details.
        **kwargs: Additional fields to include in the error response.
        
    Returns:
        A FastAPI JSONResponse with the error details.
    """
    error_data = {
        "status": "error",
        "error": {
            "code": status_code,
            "message": message,
            "type": error_type or "api_error"
        },
        **kwargs
    }
    
    if details:
        error_data["error"]["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=error_data
    )

def handle_exception(exception: Exception) -> JSONResponse:
    """Handle an exception and return an appropriate API response.
    
    Args:
        exception: The exception to handle.
        
    Returns:
        A FastAPI JSONResponse with the error details.
    """
    # Log the exception
    logger.error(f"Unhandled exception: {str(exception)}", exc_info=True)
    
    # Handle our custom API errors
    if isinstance(exception, APIError):
        return create_error_response(
            message=exception.message,
            status_code=exception.status_code,
            error_type=exception.error_type,
            details=exception.details,
            **exception.extra
        )
    
    # Handle validation errors
    if hasattr(exception, "detail"):
        # Handle FastAPI validation errors
        if isinstance(exception.detail, list):
            errors = [{"field": ".".join(str(loc) for loc in e["loc"]), "msg": e["msg"]} 
                     for e in exception.detail]
            return create_error_response(
                message="Validation error",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_type="validation_error",
                details={"errors": errors}
            )
        # Handle other FastAPI errors
        return create_error_response(
            message=str(exception.detail),
            status_code=getattr(exception, "status_code", status.HTTP_400_BAD_REQUEST)
        )
    
    # Handle generic exceptions
    return create_error_response(
        message="An unexpected error occurred",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="server_error"
    )

def validate_request(
    data: Dict[str, Any], 
    model: Type[BaseModel],
    exclude_unset: bool = False,
    **kwargs
) -> BaseModel:
    """Validate request data against a Pydantic model.
    
    Args:
        data: The data to validate.
        model: The Pydantic model to validate against.
        exclude_unset: Whether to exclude unset fields from the result.
        **kwargs: Additional arguments to pass to the model's parse_obj method.
        
    Returns:
        A validated instance of the model.
        
    Raises:
        ValidationError: If the data is invalid.
    """
    try:
        if exclude_unset:
            return model.parse_obj({k: v for k, v in data.items() if v is not None}, **kwargs)
        return model.parse_obj(data, **kwargs)
    except Exception as e:
        if hasattr(e, "errors"):
            errors = [{"field": ".".join(str(loc) for loc in err["loc"]), "msg": err["msg"]} 
                     for err in e.errors()]
            raise ValidationError(
                message="Invalid request data",
                details={"errors": errors}
            ) from e
        raise ValidationError(
            message=str(e),
            details={"error": str(e)}
        ) from e
