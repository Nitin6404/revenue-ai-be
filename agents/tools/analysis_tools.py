""
Analysis tools for the LangChain agent.

This module provides tools for data analysis, forecasting, and visualization.
"""
from typing import Dict, Any, List, Optional, Type, Union
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
from langchain.tools import BaseTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

# Configure logging
logger = logging.getLogger(__name__)

class ForecastToolInput(BaseModel):
    """Input model for the forecast tool."""
    metric: str = Field(..., description="The metric to forecast (e.g., 'revenue', 'sales')")
    periods: int = Field(12, description="Number of periods to forecast")
    freq: str = Field("M", description="Frequency of the time series ('D' for daily, 'M' for monthly, etc.)")
    group_by: Optional[str] = Field(None, description="Column to group by for multiple time series")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply to the data")

class ForecastTool(BaseTool):
    """Tool for generating time series forecasts."""
    name = "forecast"
    description = """
    Use this tool to generate time series forecasts using the Prophet model.
    Input should be a JSON string with 'metric', 'periods', 'freq', 'group_by', and 'filters'.
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
            from ..forecast_agent import forecast_revenue
            
            # Generate the forecast
            result = forecast_revenue(
                periods=input_data.get("periods", 12),
                freq=input_data.get("freq", "M"),
                metric=input_data.get("metric", "revenue"),
                group_by=input_data.get("group_by"),
                filters=input_data.get("filters", {})
            )
            
            return json.dumps({
                "status": "success",
                "data": result,
                "metadata": {
                    "metric": input_data.get("metric"),
                    "periods": input_data.get("periods", 12),
                    "freq": input_data.get("freq", "M")
                }
            })
            
        except Exception as e:
            logger.error(f"Error in forecast tool: {str(e)}", exc_info=True)
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
    
    args_schema: Type[BaseModel] = ForecastToolInput

class VisualizeDataToolInput(BaseModel):
    """Input model for the visualize_data tool."""
    data: Union[str, List[Dict], pd.DataFrame] = Field(..., description="Data to visualize (can be JSON string, list of dicts, or DataFrame)")
    chart_type: str = Field("auto", description="Type of chart to generate (auto, line, bar, pie, scatter, etc.)")
    x_axis: Optional[str] = Field(None, description="Column to use for the x-axis")
    y_axis: Optional[str] = Field(None, description="Column(s) to use for the y-axis")
    title: Optional[str] = Field(None, description="Chart title")
    x_label: Optional[str] = Field(None, description="X-axis label")
    y_label: Optional[str] = Field(None, description="Y-axis label")
    
    @validator('data', pre=True)
    def parse_data(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

class VisualizeDataTool(BaseTool):
    """Tool for generating visualizations from data."""
    name = "visualize_data"
    description = """
    Use this tool to create visualizations from data.
    Input should be a JSON string with 'data', 'chart_type', and other visualization parameters.
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
            from ..visualizer import get_visualization
            
            # Prepare the data
            data = input_data.get("data", [])
            
            # Create a mock parsed_query for the visualizer
            parsed_query = {
                "intent": "visualize",
                "chart_type": input_data.get("chart_type", "auto")
            }
            
            # Generate the visualization
            result = get_visualization(
                parsed_query=parsed_query,
                data=data
            )
            
            return json.dumps({
                "status": "success",
                "visualization": result,
                "metadata": {
                    "chart_type": input_data.get("chart_type", "auto"),
                    "data_points": len(data) if hasattr(data, '__len__') else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Error in visualize_data tool: {str(e)}", exc_info=True)
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
    
    args_schema: Type[BaseModel] = VisualizeDataToolInput

# Export tools for easy access
ANALYSIS_TOOLS = [
    ForecastTool(),
    VisualizeDataTool()
]

def get_analysis_tools() -> List[BaseTool]:
    """Get all analysis tools."""
    return ANALYSIS_TOOLS
