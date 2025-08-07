import os
import openai
import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def _format_date(date_str: str) -> str:
    try:
        date = datetime.strptime(str(date_str), '%Y-%m-%d')
        return date.strftime('%b %Y')
    except (ValueError, TypeError):
        return str(date_str)


def _format_currency(value: float, unit: str = '') -> str:
    if value >= 1e7:
        return f'\u20b9{value/1e7:.2f} Cr'
    elif value >= 1e5:
        return f'\u20b9{value/1e5:.2f} L'
    else:
        return f'\u20b9{value:,.2f}'


def get_visualization(parsed_query: Dict[str, Any], data: List[Dict]) -> Dict[str, Any]:
    if not data:
        return {"type": "text", "message": "No data available for visualization."}

    intent = parsed_query.get("intent", "")
    df = pd.DataFrame(data)

    if intent == "compare_revenue":
        df = df.sort_values(by=['year', 'quarter'])
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
            }]
        }

    elif intent == "forecast_revenue":
        if 'ds' not in df.columns or 'yhat' not in df.columns:
            return {"type": "text", "message": "Forecast data format not recognized."}
        df = df.sort_values(by='ds')
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
            }]
        }

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
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

    logger.info("Generating table visualization")
    return {
        "type": "table",
        "data": data,
        "columns": list(data[0].keys()) if data else []
    }


def get_insight(parsed_query: Dict[str, Any], data: List[Dict]) -> str:
    if not data:
        return "No data available to generate insights."

    intent = parsed_query.get("intent", "")

    if intent == "compare_revenue":
        if len(data) < 2:
            return "Not enough data points for comparison."
        sorted_data = sorted(data, key=lambda x: (x['year'], x['quarter']))
        first = sorted_data[0]
        last = sorted_data[-1]
        change_pct = ((last['revenue'] - first['revenue']) / first['revenue']) * 100
        return f"Revenue changed by {change_pct:.1f}% from {first['year']} Q{first['quarter']} to {last['year']} Q{last['quarter']}."

    elif intent == "forecast_revenue":
        values = [d['yhat'] for d in data]
        avg_forecast = sum(values) / len(values)
        min_forecast = min(values)
        max_forecast = max(values)
        return (
            f"Forecasted revenue ranges from \u20b9{min_forecast/1e7:.2f} Cr to \u20b9{max_forecast/1e7:.2f} Cr "
            f"with an average of \u20b9{avg_forecast/1e7:.2f} Cr over the next {len(data)} periods."
        )

    try:
        prompt = (
            f"You are a data analyst. Provide a concise insight about this data based on the user's query. "
            f"User's intent: {parsed_query.get('intent', 'general analysis')}. "
            f"Data: {str(data[:10])}. Keep it under 2 sentences and focus on the most important insight."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst that provides concise insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating insight with OpenAI: {str(e)}")
        return "Here's an overview of your data."
