# db/seed_csv_data.py

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from model import Base, Revenue
import os
from dotenv import load_dotenv

# Load .env for DB URL
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Read CSV
df = pd.read_csv("revenue_dashboard_data.csv")

# Setup DB engine
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)  # Create table if not exists

# Insert into DB
with engine.connect() as connection:
    with Session(bind=connection) as session:
        for _, row in df.iterrows():
            revenue_row = Revenue(
                year=int(row['Year']),
                month=row['Month'],
                department=row['Department'],
                revenue=float(row['Revenue'])
            )
            session.add(revenue_row)
        session.commit()

print("✅ Data seeded successfully!")
