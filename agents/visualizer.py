from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

def _format_date(date_str: str) -> str:
    """Format date string to a more readable format."""
    try:
        date = datetime.strptime(str(date_str), '%Y-%m-%d')
        return date.strftime('%b %Y')
    except (ValueError, TypeError):
        return str(date_str)

def _format_currency(value: float, unit: str = '') -> str:
    """Format currency values with appropriate units."""
    if value >= 1e7:  # Crores
        return f'₹{value/1e7:.2f} Cr'
    elif value >= 1e5:  # Lakhs
        return f'₹{value/1e5:.2f} L'
    else:
        return f'₹{value:,.2f}'

def get_visualization(parsed_query: Dict[str, Any], data: List[Dict]) -> Dict[str, Any]:
    """
    Generate visualization configuration based on query and data.
    
    Args:
        parsed_query: The parsed query from the user
        data: The data to visualize
        
    Returns:
        Dict containing visualization configuration
    """
    if not data:
        return {"type": "text", "message": "No data available for visualization."}
    
    intent = parsed_query.get("intent", "")
    
    # Convert data to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    if intent == "compare_revenue":
        # Sort by year and quarter for proper ordering
        df = df.sort_values(by=['year', 'quarter'])
        
        # Create labels and values
        labels = [f"Q{row['quarter']} {int(row['year'])}" for _, row in df.iterrows()]
        values = df['revenue'].tolist()
        
        return {
            "type": "bar",
            "title": "Quarterly Revenue Comparison",
            "labels": labels,
            "datasets": [{
                "label": "Revenue",
                "data": values,
                "backgroundColor": "rgba(54, 162, 235, 0.6)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "borderWidth": 1
            }],
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Revenue (₹ Cr)"
                        },
                        "ticks": {
                            "callback": f"function(value) {{return _formatCurrency(value, '₹', 'Cr');}}"
                        }
                    }
                },
                "plugins": {
                    "tooltip": {
                        "callbacks": {
                            "label": f"function(context) {{return ' ' + _formatCurrency(context.parsed.y, '₹', 'Cr');}}"
                        }
                    }
                }
            }
        }

    elif intent == "forecast_revenue":
        # Ensure we have the required columns
        if 'ds' not in df.columns or 'yhat' not in df.columns:
            return {"type": "text", "message": "Forecast data format not recognized."}
            
        # Sort by date
        df = df.sort_values(by='ds')
        
        # Format dates for display
        labels = [_format_date(ds) for ds in df['ds']]
        
        return {
            "type": "line",
            "title": "Revenue Forecast",
            "labels": labels,
            "datasets": [{
                "label": "Forecasted Revenue",
                "data": df['yhat'].tolist(),
                "fill": False,
                "borderColor": "rgb(75, 192, 192)",
                "tension": 0.1
            }],
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": False,
                        "title": {
                            "display": True,
                            "text": "Revenue (₹ Cr)"
                        },
                        "ticks": {
                            "callback": f"function(value) {{return _formatCurrency(value, '₹', 'Cr');}}"
                        }
                    }
                },
                "plugins": {
                    "tooltip": {
                        "callbacks": {
                            "label": f"function(context) {{return ' ' + _formatCurrency(context.parsed.y, '₹', 'Cr');}}"
                        }
                    }
                }
            }
        }
    
    # Default visualization for other intents
    elif len(data) > 0:
        # Try to create a simple bar chart if we have numeric data
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 0:
            # Use the first numeric column as values
            value_col = numeric_cols[0]
            label_col = next((col for col in ['name', 'label', 'id'] if col in df.columns), None)
            
            if label_col:
                df = df.sort_values(by=value_col, ascending=False)
                labels = df[label_col].astype(str).tolist()
            else:
                labels = [str(i) for i in range(len(df))]
                
            return {
                "type": "bar",
                "title": f"{value_col.replace('_', ' ').title()}",
                "labels": labels,
                "datasets": [{
                    "label": value_col.replace('_', ' ').title(),
                    "data": df[value_col].tolist(),
                    "backgroundColor": "rgba(75, 192, 192, 0.6)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "borderWidth": 1
                }]
            }
    
    # Fallback to table view
    return {
        "type": "table",
        "data": data,
        "columns": list(data[0].keys()) if data else []
    }
