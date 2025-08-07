from typing import Dict, Any, List, Optional
import os
import json
import re
import logging
from datetime import datetime, timedelta
from enum import Enum
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class TimeFrame(str, Enum):
    """Supported time frame values"""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    YTD = "ytd"
    MTD = "mtd"
    QTD = "qtd"

class Intent(str, Enum):
    """Supported query intents"""
    COMPARE = "compare"
    TREND = "trend"
    FORECAST = "forecast"
    SUMMARY = "summary"
    ANOMALY = "anomaly"
    RANK = "rank"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"

class QueryParser:
    """Parses natural language queries into structured data"""
    
    @staticmethod
    def _parse_time_frame(time_str: str) -> Dict[str, Any]:
        """
        Parse a time frame string into start and end dates.
        
        Args:
            time_str: Time frame string (e.g., 'last 3 months', 'Q2 2023')
            
        Returns:
            Dictionary with 'start' and 'end' date strings
        """
        today = datetime.now()
        time_str = time_str.lower().strip()
        
        # Handle relative time frames
        if time_str == "today":
            return {"start": today.strftime("%Y-%m-%d"), "end": today.strftime("%Y-%m-%d")}
            
        if time_str == "yesterday":
            yesterday = today - timedelta(days=1)
            return {"start": yesterday.strftime("%Y-%m-%d"), "end": yesterday.strftime("%Y-%m-%d")}
            
        if time_str == "last week":
            start = today - timedelta(days=today.weekday() + 7)
            end = start + timedelta(days=6)
            return {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}
            
        if time_str == "this month":
            start = today.replace(day=1)
            return {"start": start.strftime("%Y-%m-%d"), "end": today.strftime("%Y-%m-%d")}
            
        if time_str == "last month":
            first_day = today.replace(day=1)
            last_month = first_day - timedelta(days=1)
            first_day_last_month = last_month.replace(day=1)
            return {
                "start": first_day_last_month.strftime("%Y-%m-%d"),
                "end": last_month.strftime("%Y-%m-%d")
            }
            
        # Handle quarter format (e.g., 'Q2 2023')
        quarter_match = re.match(r'q([1-4])\s*(\d{4})?', time_str, re.IGNORECASE)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            year = int(quarter_match.group(2)) if quarter_match.group(2) else today.year
            
            if quarter == 1:
                start_month, end_month = 1, 3
            elif quarter == 2:
                start_month, end_month = 4, 6
            elif quarter == 3:
                start_month, end_month = 7, 9
            else:  # Q4
                start_month, end_month = 10, 12
                
            start_date = datetime(year, start_month, 1)
            
            # Calculate last day of end month
            if end_month == 12:
                end_date = datetime(year, 12, 31)
            else:
                end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
                
            return {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            }
            
        # Default: return as is and let the database handle it
        return {"value": time_str}
    
    @classmethod
    def parse_query(cls, user_query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into a structured format.
        
        Args:
            user_query: The natural language query from the user
            
        Returns:
            Dictionary containing the parsed query structure
            
        Raises:
            ValueError: If the query cannot be parsed
        """
        if not user_query or not isinstance(user_query, str):
            raise ValueError("Query must be a non-empty string")
            
        # Clean the query
        user_query = user_query.strip()
        
        try:
            # Try to use GPT for complex parsing
            return cls._parse_with_gpt(user_query)
        except Exception as e:
            logger.warning(f"GPT parsing failed, falling back to simple parser: {str(e)}")
            return cls._simple_parse(user_query)
    
    @classmethod
    def _parse_with_gpt(cls, user_query: str) -> Dict[str, Any]:
        """Use GPT to parse complex natural language queries"""
        system_prompt = """
        You are an advanced query parser that converts natural language into structured JSON.
        Extract the following information from the user's query:
        - intent: The main purpose of the query (e.g., 'compare', 'trend', 'forecast')
        - metrics: List of metrics being analyzed (e.g., 'revenue', 'sales_count')
        - dimensions: List of dimensions to group by (e.g., 'region', 'product_category')
        - time_frame: Time period for the analysis
        - filters: Any filters to apply to the data
        - limit: Maximum number of results to return (if specified)
        - sort: How to sort the results (if specified)
        
        Respond with a valid JSON object only.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Extract and parse the JSON response
            content = response.choices[0].message['content'].strip()
            
            # Sometimes the response includes markdown code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].strip()
                if content.startswith('json\n'):
                    content = content[5:]
            
            parsed = json.loads(content)
            
            # Validate the parsed structure
            return cls._validate_parsed_query(parsed)
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse GPT response: {str(e)}")
            raise ValueError("Failed to parse the query. Please try rephrasing.")
    
    @classmethod
    def _simple_parse(cls, user_query: str) -> Dict[str, Any]:
        """Simple keyword-based parser as a fallback"""
        query_lower = user_query.lower()
        parsed = {
            "intent": "query",
            "metrics": [],
            "dimensions": [],
            "time_frame": {},
            "filters": {}  # Add filters to the simple parser
        }
        
        # Detect metrics
        metrics = []
        for metric in ["revenue", "sales", "profit", "customers", "orders"]:
            if metric in query_lower:
                metrics.append(metric)
        
        if metrics:
            parsed["metrics"] = metrics
        
        # Detect time frames
        time_phrases = {
            "today": "today",
            "yesterday": "yesterday",
            "week": "last 7 days",
            "month": "this month",
            "quarter": "this quarter",
            "year": "this year",
            "ytd": "year to date"
        }
        
        for phrase, time_frame in time_phrases.items():
            if phrase in query_lower:
                parsed["time_frame"] = cls._parse_time_frame(time_frame)
                break
        
        # Detect filters
        filters = []
        for filter in ["region", "product_category", "country"]:
            if filter in query_lower:
                filters.append(filter)
        
        if filters:
            parsed["filters"] = filters
        
        return parsed
    
    @staticmethod
    def _validate_parsed_query(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the parsed query"""
        # Ensure required fields exist
        if not parsed.get("metrics"):
            raise ValueError("No metrics specified in the query")
        
        # Convert to list if single metric provided as string
        if isinstance(parsed["metrics"], str):
            parsed["metrics"] = [parsed["metrics"]]
            
        # Normalize intent
        if "intent" in parsed:
            parsed["intent"] = parsed["intent"].lower()
            
        # Normalize time_frame
        if "time_frame" in parsed and isinstance(parsed["time_frame"], str):
            parsed["time_frame"] = {"value": parsed["time_frame"]}
            
        # Ensure dimensions is a list
        if "dimensions" not in parsed:
            parsed["dimensions"] = []
        elif isinstance(parsed["dimensions"], str):
            parsed["dimensions"] = [parsed["dimensions"]]
            
        return parsed

# Alias for backward compatibility
def parse_query(user_query: str) -> Dict[str, Any]:
    return QueryParser.parse_query(user_query)
