from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from prophet import Prophet
from db.database import engine
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_time_series_data(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    Prepare time series data for forecasting.
    
    Args:
        df: Input DataFrame with time series data
        date_col: Name of the date column
        value_col: Name of the value column to forecast
        
    Returns:
        DataFrame with 'ds' and 'y' columns ready for Prophet
    """
    # Ensure proper date format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Handle missing dates by resampling
    df = df.set_index(date_col).resample('D').sum().reset_index()
    
    # Prepare for Prophet
    df = df.rename(columns={date_col: 'ds', value_col: 'y'})
    
    return df

def forecast_revenue(
    periods: int = 12,
    freq: str = 'M',
    metric: str = 'revenue',
    group_by: Optional[str] = None,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    Generate revenue forecast using Facebook's Prophet.
    
    Args:
        periods: Number of periods to forecast
        freq: Frequency of the time series ('D' for daily, 'M' for monthly, etc.)
        metric: The metric to forecast (e.g., 'revenue', 'sales')
        group_by: Optional column to group by for multiple time series
        filters: Optional dictionary of filters to apply to the data
        
    Returns:
        List of dictionaries containing forecast results
    """
    try:
        # Build the base query
        select_columns = ["date", f"SUM({metric}) AS {metric}"]
        group_by_clause = ["date"]
        
        if group_by:
            select_columns.append(group_by)
            group_by_clause.append(group_by)
        
        query = f"""
            SELECT {', '.join(select_columns)}
            FROM revenue
            {f"WHERE {' AND '.join([f"{k} = '{v}'" for k, v in (filters or {}).items()])}" if filters else ""}
            GROUP BY {', '.join(group_by_clause)}
            ORDER BY date ASC
        """
        
        logger.info(f"Executing query: {query}")
        
        # Fetch historical data
        df = pd.read_sql_query(query, con=engine)
        
        if df.empty:
            logger.warning("No data found for the given filters")
            return []
            
        # Handle multiple time series if group_by is specified
        if group_by and group_by in df.columns:
            results = []
            for group_name, group_df in df.groupby(group_by):
                if len(group_df) < 2:  # Need at least 2 points for forecasting
                    logger.warning(f"Not enough data points for group: {group_name}")
                    continue
                    
                # Prepare data for this group
                ts_data = prepare_time_series_data(group_df, 'date', metric)
                
                # Generate forecast
                forecast_df = _generate_forecast(ts_data, periods, freq)
                
                # Add group information
                forecast_df[group_by] = group_name
                results.append(forecast_df)
                
            if not results:
                return []
                
            # Combine results from all groups
            result_df = pd.concat(results, ignore_index=True)
        else:
            # Single time series
            if len(df) < 2:
                logger.warning("Not enough data points for forecasting")
                return []
                
            ts_data = prepare_time_series_data(df, 'date', metric)
            result_df = _generate_forecast(ts_data, periods, freq)
        
        # Convert to list of dicts and format dates
        result_df['ds'] = result_df['ds'].dt.strftime('%Y-%m-%d')
        return result_df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error in forecast_revenue: {str(e)}", exc_info=True)
        raise

def _generate_forecast(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """
    Internal function to generate forecast using Prophet.
    
    Args:
        df: DataFrame with 'ds' and 'y' columns
        periods: Number of periods to forecast
        freq: Frequency of the time series
        
    Returns:
        DataFrame with forecast results
    """
    try:
        # Initialize and fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Add custom seasonality for business cycles
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit model
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq, include_history=False)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Return only the forecasted values
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
    except Exception as e:
        logger.error(f"Error in _generate_forecast: {str(e)}")
        raise
