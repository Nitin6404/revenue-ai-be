"""
Data-related tools for the LangChain agent.

This module provides tools for interacting with the database and performing data operations.
"""
from typing import Dict, Any, List, Optional, Type, Union
import json
import logging
import pandas as pd
from pydantic import BaseModel, Field, validator
from langchain.tools import BaseTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

# Configure logging
logger = logging.getLogger(__name__)

class DataToolInput(BaseModel):
    """Base input model for data tools."""
    query: str = Field(..., description="The query or parameters for the tool")

class FetchDataToolInput(DataToolInput):
    """Input model for the fetch_data tool."""
    sql: str = Field(..., description="SQL query to execute")
    params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    return_df: bool = Field(False, description="Whether to return a pandas DataFrame")

class FetchDataTool(BaseTool):
    """Tool for fetching data from the database."""
    name = "fetch_data"
    description = """
    Use this tool to execute a SQL query against the database.
    Input should be a JSON string with 'sql', 'params', and 'return_df' keys.
    """
    
    def _run(
        self, 
        tool_input: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool synchronously."""
        try:
            # Parse the input
            input_data = json.loads(tool_input)
            sql = input_data.get("sql")
            params = input_data.get("params", {})
            return_df = input_data.get("return_df", False)
            
            if not sql:
                raise ValueError("SQL query is required")
                
            # Import here to avoid circular imports
            from ..data_fetcher import run_sql
            
            # Execute the query
            result = run_sql(sql, params, return_df=return_df)
            
            # Convert to JSON-serializable format
            if return_df and isinstance(result, pd.DataFrame):
                result_data = result.to_dict(orient="records")
            elif isinstance(result, (list, dict)):
                result_data = result
            else:
                result_data = str(result)
            
            return json.dumps({
                "status": "success",
                "data": result_data
            })
            
        except Exception as e:
            logger.error(f"Error in fetch_data tool: {str(e)}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    async def _arun(
        self, 
        tool_input: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool asynchronously."""
        return self._run(tool_input, run_manager)
    
    args_schema: Type[BaseModel] = FetchDataToolInput

class GetMetricToolInput(DataToolInput):
    """Input model for the get_metric tool."""
    metric: str = Field(..., description="The metric to fetch (e.g., 'revenue', 'sales_count')")
    dimensions: Optional[List[str]] = Field(
        None, 
        description="List of dimensions to group by (e.g., ['region', 'product_category'])"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, 
        description="Dictionary of filter conditions (e.g., {'date >': '2023-01-01'})"
    )
    limit: Optional[int] = Field(
        None, 
        description="Maximum number of rows to return"
    )

class GetMetricTool(BaseTool):
    """Tool for fetching metrics with flexible grouping and filtering."""
    name = "get_metric"
    description = """
    Use this tool to fetch a specific metric with optional grouping and filtering.
    Input should be a JSON string with 'metric', 'dimensions', 'filters', and 'limit'.
    """
    
    def _run(
        self, 
        tool_input: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool synchronously."""
        try:
            # Parse the input
            input_data = json.loads(tool_input)
            
            # Import here to avoid circular imports
            from ..data_fetcher import fetch_metric
            
            # Fetch the data
            result = fetch_metric(
                metric=input_data.get("metric"),
                dimensions=input_data.get("dimensions"),
                filters=input_data.get("filters", {}),
                limit=input_data.get("limit")
            )
            
            return json.dumps({
                "status": "success",
                "data": result,
                "metadata": {
                    "metric": input_data.get("metric"),
                    "dimensions": input_data.get("dimensions", []),
                    "filter_count": len(input_data.get("filters", {})),
                    "row_count": len(result) if hasattr(result, '__len__') else 1
                }
            })
            
        except Exception as e:
            logger.error(f"Error in get_metric tool: {str(e)}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    async def _arun(
        self, 
        tool_input: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool asynchronously."""
        return self._run(tool_input, run_manager)
    
    args_schema: Type[BaseModel] = GetMetricToolInput

class ListTablesTool(BaseTool):
    """Tool for listing available database tables."""
    name = "list_tables"
    description = "List all available tables in the database."
    
    def _run(
        self, 
        tool_input: str = "", 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool synchronously."""
        try:
            # Import here to avoid circular imports
            from sqlalchemy import inspect
            from ..data_fetcher import engine
            
            # Get table information
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            # Get column information for each table
            table_info = []
            for table_name in tables:
                columns = [col["name"] for col in inspector.get_columns(table_name)]
                table_info.append({
                    "table_name": table_name,
                    "columns": columns
                })
            
            return json.dumps({
                "status": "success",
                "tables": table_info
            })
            
        except Exception as e:
            logger.error(f"Error in list_tables tool: {str(e)}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    async def _arun(
        self, 
        tool_input: str = "", 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool asynchronously."""
        return self._run(tool_input, run_manager)

# Export tools for easy access
DATA_TOOLS = [
    FetchDataTool(),
    GetMetricTool(),
    ListTablesTool()
]

def get_data_tools() -> List[BaseTool]:
    """Get all data tools."""
    return DATA_TOOLS
