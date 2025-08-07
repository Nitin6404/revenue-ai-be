"""
Tools package for the LangChain agent.

This package contains all the tools available to the LangChain agent.
"""
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool

# Import all tool modules
from .data_tools import get_data_tools
from .analysis_tools import get_analysis_tools
from .insight_tools import get_insight_tools

def get_all_tools() -> List[BaseTool]:
    """
    Get all available tools for the LangChain agent.
    
    Returns:
        List[BaseTool]: A list of all available tools.
    """
    tools = []
    tools.extend(get_data_tools())
    tools.extend(get_analysis_tools())
    tools.extend(get_insight_tools())
    return tools

def get_tool_by_name(tool_name: str) -> Optional[BaseTool]:
    """
    Get a tool by its name.
    
    Args:
        tool_name: The name of the tool to retrieve.
        
    Returns:
        Optional[BaseTool]: The tool with the specified name, or None if not found.
    """
    for tool in get_all_tools():
        if tool.name == tool_name:
            return tool
    return None

def get_tool_descriptions() -> List[Dict[str, Any]]:
    """
    Get descriptions of all available tools.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing tool names and descriptions.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "args_schema": str(tool.args_schema.schema() if hasattr(tool, 'args_schema') else {})
        }
        for tool in get_all_tools()
    ]
