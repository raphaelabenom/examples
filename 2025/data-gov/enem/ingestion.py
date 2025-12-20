import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL").replace(
    "postgresql://", "postgresql+psycopg2://"
)

base = os.getcwd()
path_tables = os.path.join(base, "tables")
tables = os.listdir(path_tables)

con = create_engine(DATABASE_URL)

for table in tables:
    path_table = os.path.join(path_tables, table)
    table = table.replace(".csv", "")
    df = pd.read_csv(path_table, encoding="utf-8")
    df.to_sql(table, con, schema="public", if_exists="replace", index=False)
    print(f"Table {table} uploaded!")
