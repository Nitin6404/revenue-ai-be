
from utils.time_utils import get_date_range

def generate_sql(parsed_query: dict):
    if parsed_query["intent"] == "compare_revenue":
        current_range = get_date_range(parsed_query["time_frame"]["current"])
        previous_range = get_date_range(parsed_query["time_frame"]["previous"])

        return f"""
        SELECT
          EXTRACT(YEAR FROM date) AS year,
          EXTRACT(QUARTER FROM date) AS quarter,
          SUM(amount) AS revenue
        FROM revenue
        WHERE (date BETWEEN '{previous_range[0]}' AND '{previous_range[1]}')
           OR (date BETWEEN '{current_range[0]}' AND '{current_range[1]}')
        GROUP BY year, quarter;
        """
    raise NotImplementedError("Other intents not handled yet")
