
from datetime import datetime

def get_date_range(label):
    # Simplified; later make dynamic
    if label == "2025-Q2":
        return ("2025-04-01", "2025-06-30")
    elif label == "2024-Q2":
        return ("2024-04-01", "2024-06-30")
    else:
        raise ValueError("Unknown time frame")
