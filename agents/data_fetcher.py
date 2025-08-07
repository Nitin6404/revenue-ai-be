
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from db.database import engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryError(Exception):
    """Custom exception for query execution errors"""
    pass

def run_sql(
    query: str, 
    params: Optional[Dict[str, Any]] = None,
    return_df: bool = False
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
    Execute a SQL query and return the results where our table name is revenue.
    
    Args:
        query: SQL query to execute
        params: Dictionary of parameters for parameterized queries
        return_df: If True, return a pandas DataFrame instead of a list of dicts
        
    Returns:
        List of dictionaries or a pandas DataFrame with the query results
        
    Raises:
        QueryError: If there's an error executing the query
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    
    params = params or {}
    
    try:
        # Log the query (without parameters for security)
        logger.info(f"Executing SQL query: {query[:200]}...")
        
        # Use SQLAlchemy's text() for safe parameter binding
        with engine.connect() as connection:
            result = connection.execute(text(query), params)
            
            # Convert to DataFrame first for better type handling
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            # Convert date/datetime objects to ISO format strings
            for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
                df[col] = df[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
            
            # Return either DataFrame or list of dicts based on return_df flag
            if return_df:
                return df
            return df.to_dict(orient='records')
            
    except SQLAlchemyError as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise QueryError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error executing query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise QueryError(error_msg) from e

def fetch_metric(
    metric: str,
    dimensions: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[Union[str, List[str]]] = None,
    limit: Optional[int] = None,
    return_df: bool = False
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
    Fetch metric data with optional filtering, grouping, and sorting.
    
    Args:
        metric: The metric to fetch (e.g., 'revenue', 'sales_count')
        dimensions: List of dimensions to group by (e.g., ['date', 'region'])
        filters: Dictionary of filter conditions (e.g., {'date >': '2023-01-01'})
        order_by: Column(s) to order results by
        limit: Maximum number of rows to return
        return_df: If True, return a pandas DataFrame
        
    Returns:
        Query results as a list of dictionaries or a DataFrame
    """
    try:
        # Build SELECT clause
        select_clause = [f"SUM({metric}) AS {metric}"]
        
        # Add dimensions to SELECT and GROUP BY
        group_by_clause = []
        if dimensions:
            select_clause.extend(dimensions)
            group_by_clause = dimensions.copy()
        
        # Build WHERE clause
        where_conditions = []
        params = {}
        
        if filters:
            for i, (col, val) in enumerate(filters.items()):
                param_name = f"param_{i}"
                if ' ' in col:
                    # Handle operators in column names (e.g., 'date >')
                    col_name, op = col.rsplit(' ', 1)
                    where_conditions.append(f"{col_name} {op} :{param_name}")
                else:
                    where_conditions.append(f"{col} = :{param_name}")
                params[param_name] = val
        
        # Build ORDER BY clause
        order_by_clause = ""
        if order_by:
            if isinstance(order_by, str):
                order_by = [order_by]
            order_by_clause = f"ORDER BY {', '.join(order_by)}"
        
        # Build LIMIT clause
        limit_clause = f"LIMIT {limit}" if limit is not None else ""
        
        # Build GROUP BY clause
        group_by_str = f"GROUP BY {', '.join(group_by_clause)}" if group_by_clause else ""
        
        # Build the full query
        query = f"""
            SELECT {', '.join(select_clause)}
            FROM revenue
            {f'WHERE {' AND '.join(where_conditions)}' if where_conditions else ''}
            {group_by_str}
            {order_by_clause}
            {limit_clause}
        """.strip()
        
        # Execute the query
        return run_sql(query, params, return_df)
        
    except Exception as e:
        error_msg = f"Error fetching metric '{metric}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise QueryError(error_msg) from e
