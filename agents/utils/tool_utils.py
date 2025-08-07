""" 
Utility functions for working with LangChain tools.

This module provides helper functions for serializing, deserializing, and working with LangChain tools.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError
from langchain.tools import BaseTool, Tool

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)

def serialize_tool_input(tool_input: Union[BaseModel, Dict[str, Any]]) -> str:
    """
    Serialize a tool input to a JSON string.
    
    Args:
        tool_input: The tool input to serialize, either a Pydantic model or a dictionary.
        
    Returns:
        str: A JSON string representation of the tool input.
    """
    if isinstance(tool_input, BaseModel):
        return tool_input.json()
    return json.dumps(tool_input)

def deserialize_tool_input(
    tool_input: str, 
    model: Type[T]
) -> T:
    """
    Deserialize a JSON string into a Pydantic model.
    
    Args:
        tool_input: The JSON string to deserialize.
        model: The Pydantic model class to deserialize into.
        
    Returns:
        An instance of the specified Pydantic model.
        
    Raises:
        ValueError: If the input cannot be deserialized into the specified model.
    """
    try:
        if isinstance(tool_input, str):
            data = json.loads(tool_input)
        else:
            data = tool_input
            
        if isinstance(data, dict):
            return model(**data)
        return model(data)  # type: ignore
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        error_msg = f"Failed to deserialize tool input: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e

def get_tool_by_name(tools: List[BaseTool], tool_name: str) -> Optional[BaseTool]:
    """
    Find a tool by its name in a list of tools.
    
    Args:
        tools: List of tools to search through.
        tool_name: Name of the tool to find.
        
    Returns:
        The tool with the specified name, or None if not found.
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None

def validate_tool_input(
    tool: BaseTool, 
    tool_input: Union[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate tool input against the tool's input schema.
    
    Args:
        tool: The tool to validate input for.
        tool_input: The input to validate.
        
    Returns:
        The validated input as a dictionary.
        
    Raises:
        ValueError: If the input is invalid.
    """
    if not hasattr(tool, 'args_schema'):
        if isinstance(tool_input, str):
            try:
                return json.loads(tool_input)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON input: {str(e)}") from e
        return tool_input
    
    # Convert input to model instance for validation
    try:
        if isinstance(tool_input, str):
            input_data = json.loads(tool_input)
        else:
            input_data = tool_input
            
        # Create model instance to validate
        model = tool.args_schema
        if isinstance(model, type) and issubclass(model, BaseModel):
            return model(**input_data).dict()
        return input_data
    except (json.JSONDecodeError, ValidationError) as e:
        error_msg = f"Invalid input for tool '{tool.name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e

def format_tool_result(
    result: Any, 
    tool_name: str, 
    error: bool = False
) -> Dict[str, Any]:
    """
    Format a tool result in a standard format.
    
    Args:
        result: The result to format.
        tool_name: The name of the tool that produced the result.
        error: Whether the result is an error.
        
    Returns:
        A dictionary containing the formatted result.
    """
    if error:
        return {
            "status": "error",
            "tool": tool_name,
            "result": str(result) if not isinstance(result, dict) else result
        }
    
    return {
        "status": "success",
        "tool": tool_name,
        "result": result if isinstance(result, dict) else {"output": result}
    }

def create_tool_from_function(
    func: callable,
    name: str,
    description: str,
    args_schema: Optional[Type[BaseModel]] = None
) -> Tool:
    """
    Create a LangChain Tool from a Python function.
    
    Args:
        func: The function to wrap as a tool.
        name: The name of the tool.
        description: A description of what the tool does.
        args_schema: Optional Pydantic model for input validation.
        
    Returns:
        A LangChain Tool instance.
    """
    return Tool(
        name=name,
        description=description,
        func=func,
        args_schema=args_schema,
        return_direct=False
    )

def execute_tool(
    tool: BaseTool,
    tool_input: Union[str, Dict[str, Any]],
    **kwargs
) -> Any:
    """
    Execute a tool with the given input.
    
    Args:
        tool: The tool to execute.
        tool_input: The input to pass to the tool.
        **kwargs: Additional keyword arguments to pass to the tool's _run method.
        
    Returns:
        The result of executing the tool.
        
    Raises:
        ValueError: If the tool execution fails.
    """
    try:
        # Validate input
        validated_input = validate_tool_input(tool, tool_input)
        
        # Execute the tool
        if isinstance(tool, Tool):
            result = tool._run(validated_input, **kwargs)
        else:
            result = tool._run(validated_input, **kwargs)
            
        return format_tool_result(result, tool.name)
    except Exception as e:
        logger.error(f"Error executing tool {tool.name}: {str(e)}", exc_info=True)
        return format_tool_result(str(e), tool.name, error=True)
