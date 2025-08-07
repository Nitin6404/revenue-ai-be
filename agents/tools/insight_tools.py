"""
Insight generation tools for the LangChain agent.

This module provides tools for generating natural language insights from data.
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

class GenerateInsightToolInput(BaseModel):
    """Input model for the generate_insight tool."""
    data: Union[str, List[Dict], pd.DataFrame] = Field(
        ..., 
        description="Data to analyze (can be JSON string, list of dicts, or DataFrame)"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context for insight generation"
    )
    intent: Optional[str] = Field(
        None,
        description="The intent of the user's query (e.g., 'trend', 'comparison', 'anomaly')"
    )
    
    @validator('data', pre=True)
    def parse_data(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

class GenerateInsightTool(BaseTool):
    """Tool for generating natural language insights from data."""
    name = "generate_insight"
    description = """
    Use this tool to generate natural language insights from data.
    Input should be a JSON string with 'data' and optional 'context' and 'intent'.
    """
    
    def _run(
        self, 
        tool_input: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool synchronously."""
        try:
            # Parse the input
            if isinstance(tool_input, str):
                input_data = json.loads(tool_input)
            else:
                input_data = tool_input
            
            # Import here to avoid circular imports
            from ..insight_agent import get_insight
            
            # Prepare the data and context
            data = input_data.get("data", [])
            context = input_data.get("context", {})
            intent = input_data.get("intent", "analysis")
            
            # Generate the insight
            insight = get_insight(
                parsed_query={"intent": intent},
                data=data,
                context=context
            )
            
            return json.dumps({
                "status": "success",
                "insight": insight,
                "metadata": {
                    "intent": intent,
                    "data_points": len(data) if hasattr(data, '__len__') else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Error in generate_insight tool: {str(e)}", exc_info=True)
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
    
    args_schema: Type[BaseModel] = GenerateInsightToolInput

class CompareDataToolInput(BaseModel):
    """Input model for the compare_data tool."""
    data: Union[str, List[Dict], pd.DataFrame] = Field(
        ..., 
        description="Data to compare (can be JSON string, list of dicts, or DataFrame)"
    )
    comparison_dimension: str = Field(
        ...,
        description="The dimension to compare (e.g., 'region', 'product')"
    )
    metric: str = Field(
        ...,
        description="The metric to compare (e.g., 'revenue', 'sales')"
    )
    top_n: Optional[int] = Field(
        5,
        description="Number of top items to compare"
    )

class CompareDataTool(BaseTool):
    """Tool for comparing data across different dimensions."""
    name = "compare_data"
    description = """
    Use this tool to compare data across different dimensions (e.g., regions, products).
    Input should be a JSON string with 'data', 'comparison_dimension', 'metric', and 'top_n'.
    """
    
    def _run(
        self, 
        tool_input: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool synchronously."""
        try:
            # Parse the input
            if isinstance(tool_input, str):
                input_data = json.loads(tool_input)
            else:
                input_data = tool_input
            
            # Convert data to DataFrame if it's not already
            data = input_data.get("data", [])
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Get parameters
            dimension = input_data.get("comparison_dimension")
            metric = input_data.get("metric")
            top_n = input_data.get("top_n", 5)
            
            if dimension not in data.columns:
                raise ValueError(f"Dimension '{dimension}' not found in data")
                
            if metric not in data.columns:
                raise ValueError(f"Metric '{metric}' not found in data")
            
            # Perform the comparison
            comparison = data.groupby(dimension)[metric].agg(['sum', 'mean', 'count'])
            comparison = comparison.sort_values('sum', ascending=False).head(top_n)
            
            # Convert to dict for JSON serialization
            result = comparison.reset_index().to_dict(orient='records')
            
            return json.dumps({
                "status": "success",
                "comparison": result,
                "metadata": {
                    "dimension": dimension,
                    "metric": metric,
                    "top_n": top_n,
                    "total_items": len(comparison)
                }
            })
            
        except Exception as e:
            logger.error(f"Error in compare_data tool: {str(e)}", exc_info=True)
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
    
    args_schema: Type[BaseModel] = CompareDataToolInput

# Export tools for easy access
INSIGHT_TOOLS = [
    GenerateInsightTool(),
    CompareDataTool()
]

def get_insight_tools() -> List[BaseTool]:
    """Get all insight generation tools."""
    return INSIGHT_TOOLS
