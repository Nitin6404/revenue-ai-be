"""
Data processing and transformation utilities.

This module provides helper functions for common data manipulation tasks,
such as filtering, aggregation, and transformation of data structures.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil import parser

# Configure logging
logger = logging.getLogger(__name__)

def safe_json_loads(data: str) -> Any:
    """Safely parse a JSON string, returning the original string if parsing fails."""
    if not isinstance(data, str):
        return data
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return data

def convert_to_dataframe(
    data: Union[List[Dict], Dict, pd.DataFrame, str],
    **kwargs
) -> pd.DataFrame:
    """Convert various data formats to a pandas DataFrame."""
    if data is None:
        return pd.DataFrame()
    
    if isinstance(data, pd.DataFrame):
        return data.copy()
    
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {str(e)}") from e
    
    if isinstance(data, dict):
        if not data:  # Empty dict
            return pd.DataFrame(**kwargs)
        
        # Check if it's a single record or needs to be normalized
        if any(isinstance(v, (list, dict)) for v in data.values()):
            return pd.json_normalize(data, **kwargs)
        else:
            return pd.DataFrame([data], **kwargs)
    
    if isinstance(data, list):
        if not data:  # Empty list
            return pd.DataFrame(**kwargs)
            
        # Check if it's a list of dicts or a list of other types
        if all(isinstance(item, dict) for item in data):
            return pd.DataFrame(data, **kwargs)
        else:
            return pd.DataFrame({"value": data}, **kwargs)
    
    raise ValueError(f"Cannot convert type {type(data)} to DataFrame")

def filter_dataframe(
    df: pd.DataFrame,
    filters: Dict[str, Any],
    operator: str = "and"
) -> pd.DataFrame:
    """Filter a DataFrame based on a dictionary of column filters."""
    if df.empty or not filters:
        return df
    
    conditions = []
    
    for column, value in filters.items():
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            continue
            
        # Handle operator syntax: {"column": {"operator": value}}
        if isinstance(value, dict) and len(value) == 1:
            op, val = next(iter(value.items()))
        else:
            op, val = "eq", value
        
        # Apply the appropriate filter
        try:
            if op == "eq":
                conditions.append(df[column] == val)
            elif op == "ne":
                conditions.append(df[column] != val)
            elif op == "gt":
                conditions.append(df[column] > val)
            elif op == "lt":
                conditions.append(df[column] < val)
            elif op == "ge":
                conditions.append(df[column] >= val)
            elif op == "le":
                conditions.append(df[column] <= val)
            elif op == "in":
                if not isinstance(val, (list, tuple, set)):
                    val = [val]
                conditions.append(df[column].isin(val))
            elif op == "contains":
                conditions.append(df[column].astype(str).str.contains(str(val), case=False, na=False))
            elif op == "startswith":
                conditions.append(df[column].astype(str).str.startswith(str(val), na=False))
            elif op == "endswith":
                conditions.append(df[column].astype(str).str.endswith(str(val), na=False))
            else:
                logger.warning(f"Unsupported operator: {op}. Using equality ('eq').")
                conditions.append(df[column] == val)
        except Exception as e:
            logger.warning(f"Error applying filter {column} {op} {val}: {str(e)}")
    
    if not conditions:
        return df
    
    # Combine conditions with the specified operator
    if operator.lower() == "or":
        combined_condition = conditions[0]
        for cond in conditions[1:]:
            combined_condition = combined_condition | cond
    else:  # default to "and"
        combined_condition = conditions[0]
        for cond in conditions[1:]:
            combined_condition = combined_condition & cond
    
    return df[combined_condition].copy()

def aggregate_dataframe(
    df: pd.DataFrame,
    group_by: Union[str, List[str]],
    aggregations: Dict[str, Union[str, Dict[str, str]]],
    reset_index: bool = True
) -> pd.DataFrame:
    """Aggregate a DataFrame by one or more columns."""
    if df.empty or not group_by:
        return df
    
    # Convert group_by to list if it's a single string
    if isinstance(group_by, str):
        group_by = [group_by]
    
    # Check if all group_by columns exist
    missing_cols = [col for col in group_by if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")
    
    # Process aggregations
    agg_dict = {}
    for col, agg in aggregations.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found for aggregation")
            continue
            
        if isinstance(agg, dict):
            # Handle multiple aggregations for the same column
            for new_col, agg_func in agg.items():
                agg_dict[new_col] = pd.NamedAgg(column=col, aggfunc=agg_func)
        else:
            # Single aggregation
            agg_dict[col] = agg
    
    if not agg_dict:
        return df.groupby(group_by).size().reset_index(name='count')
    
    # Perform the aggregation
    result = df.groupby(group_by).agg(**agg_dict)
    
    if reset_index:
        return result.reset_index()
    
    return result

def normalize_dates(
    df: pd.DataFrame,
    date_columns: Optional[Union[str, List[str]]] = None,
    date_format: Optional[str] = None,
    errors: str = 'coerce'
) -> pd.DataFrame:
    """Convert date columns to datetime objects."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # If no date columns specified, try to detect them
    if date_columns is None:
        date_columns = []
        for col in df.columns:
            # Check if column name suggests it's a date
            if any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year']):
                date_columns.append(col)
    
    # Convert to list if single column name provided
    if isinstance(date_columns, str):
        date_columns = [date_columns]
    
    # Process each date column
    for col in date_columns:
        if col not in df.columns:
            continue
            
        try:
            if date_format:
                df[col] = pd.to_datetime(
                    df[col], 
                    format=date_format, 
                    errors=errors
                )
            else:
                # Try to infer the format
                df[col] = pd.to_datetime(df[col], errors=errors)
                
                # If conversion failed for all values, try with dateutil's parser
                if df[col].isna().all() and errors != 'raise':
                    df[col] = df[col].apply(
                        lambda x: parser.parse(str(x)) if pd.notna(x) else pd.NaT
                    )
        except Exception as e:
            logger.warning(f"Error converting column '{col}' to datetime: {str(e)}")
            if errors == 'raise':
                raise
    
    return df

def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect the data types of each column in a DataFrame."""
    if df.empty:
        return {}
    
    type_mapping = {
        'int64': 'integer',
        'float64': 'float',
        'bool': 'boolean',
        'datetime64[ns]': 'datetime',
        'object': 'string',
        'category': 'categorical'
    }
    
    detected_types = {}
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        
        # Handle datetime detection
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            detected_types[col] = 'datetime'
        # Handle numeric types
        elif pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]):
                detected_types[col] = 'integer'
            else:
                detected_types[col] = 'float'
        # Handle boolean types
        elif pd.api.types.is_bool_dtype(df[col]):
            detected_types[col] = 'boolean'
        # Handle categorical types
        elif pd.api.types.is_categorical_dtype(df[col]):
            detected_types[col] = 'categorical'
        # Handle string/object types
        else:
            # Check if it's actually a date string
            sample = df[col].dropna().head(100)  # Sample first 100 non-null values
            if not sample.empty:
                try:
                    pd.to_datetime(sample, errors='raise')
                    detected_types[col] = 'datetime'
                    continue
                except (ValueError, TypeError):
                    pass
            
            # Default to string
            detected_types[col] = 'string'
    
    return detected_types

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up column names in a DataFrame."""
    if df.empty:
        return df
    
    df = df.copy()
    
    def clean_name(name):
        if not isinstance(name, str):
            name = str(name)
        # Convert to lowercase and replace spaces/special chars with underscores
        name = name.lower().strip()
        name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        # Remove multiple underscores
        while '__' in name:
            name = name.replace('__', '_')
        return name
    
    df.columns = [clean_name(col) for col in df.columns]
    return df

def normalize_text_columns(
    df: pd.DataFrame,
    text_columns: Optional[Union[str, List[str]]] = None,
    case: str = 'lower',
    strip: bool = True,
    remove_special: bool = True
) -> pd.DataFrame:
    """Normalize text columns in a DataFrame."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Determine which columns to process
    if text_columns is None:
        text_columns = [col for col, dtype in df.dtypes.items() 
                       if pd.api.types.is_string_dtype(dtype)]
    elif isinstance(text_columns, str):
        text_columns = [text_columns]
    
    # Process each text column
    for col in text_columns:
        if col not in df.columns:
            continue
            
        # Convert to string and handle NaN values
        df[col] = df[col].astype(str)
        
        # Apply text transformations
        if case == 'lower':
            df[col] = df[col].str.lower()
        elif case == 'upper':
            df[col] = df[col].str.upper()
            
        if strip:
            df[col] = df[col].str.strip()
            
        if remove_special:
            df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
    
    return df

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'drop',
    fill_value: Any = None,
    columns: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """Handle missing values in a DataFrame."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Determine which columns to process
    if columns is None:
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]
    
    # Process each column
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            # Drop rows with missing values in this column
            df = df[df[col].notna()].copy()
        elif strategy == 'fill':
            # Fill missing values with the specified value
            if fill_value is None:
                # Use default fill values based on column type
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = 0
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    fill_value = pd.NaT
                else:
                    fill_value = ''
            
            df[col] = df[col].fillna(fill_value)
        # If strategy is 'ignore', do nothing
    
    return df

def sample_dataframe(
    df: pd.DataFrame,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """Randomly sample rows from a DataFrame."""
    if df.empty:
        return df
    
    if n is None and frac is None:
        # Default to 10% of the data
        frac = 0.1
    
    return df.sample(n=n, frac=frac, random_state=random_state).copy()

def describe_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate descriptive statistics for a DataFrame."""
    if df.empty:
        return {}
    
    result = {
        'shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'columns': list(df.columns),
        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': int(df.isna().sum().sum()),
        'missing_values_by_column': df.isna().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    # Add numeric statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        result['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Add categorical statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        result['categorical_stats'][col] = {
            'unique_values': int(df[col].nunique()),
            'top_values': value_counts.head().to_dict(),
            'freq': int(value_counts.iloc[0]) if not value_counts.empty else 0
        }
    
    return result

def prepare_for_serialization(df: pd.DataFrame) -> Dict[str, Any]:
    """Prepare a DataFrame for JSON serialization."""
    if df.empty:
        return {'data': [], 'columns': []}
    
    # Convert datetime columns to ISO format strings
    df = df.copy()
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = df[col].apply(
            lambda x: x.isoformat() if pd.notna(x) else None
        )
    
    # Convert to dictionary
    result = {
        'data': df.to_dict(orient='records'),
        'columns': [{'name': col, 'type': str(dtype)} 
                   for col, dtype in df.dtypes.items()]
    }
    
    return result

def merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    how: str = 'inner',
    **kwargs
) -> pd.DataFrame:
    """Merge two DataFrames."""
    if left.empty or right.empty:
        return left if right.empty else right
    
    return pd.merge(left, right, on=on, how=how, **kwargs)

def sort_dataframe(
    df: pd.DataFrame,
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True
) -> pd.DataFrame:
    """Sort a DataFrame by one or more columns."""
    if df.empty or not by:
        return df
    
    return df.sort_values(by=by, ascending=ascending).copy()

def rename_columns(
    df: pd.DataFrame,
    column_mapping: Dict[str, str]
) -> pd.DataFrame:
    """Rename columns in a DataFrame."""
    if df.empty or not column_mapping:
        return df
    
    return df.rename(columns=column_mapping).copy()

def select_columns(
    df: pd.DataFrame,
    columns: Union[str, List[str]]
) -> pd.DataFrame:
    """Select specific columns from a DataFrame."""
    if df.empty or not columns:
        return df
    
    if isinstance(columns, str):
        columns = [columns]
    
    # Only include columns that exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns]
    
    if not valid_columns:
        return pd.DataFrame()
    
    return df[valid_columns].copy()

def apply_transformation(
    df: pd.DataFrame,
    column: str,
    func: Callable,
    output_column: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Apply a transformation function to a column."""
    if df.empty or column not in df.columns:
        return df
    
    df = df.copy()
    output_column = output_column or column
    
    df[output_column] = df[column].apply(lambda x: func(x, **kwargs))
    
    return df
