import os
import urllib.request

import duckdb

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
PARQUET_FILE = "yellow_tripdata_2024-01.parquet"

con = duckdb.connect("taxi.db")

def setup_database():
    """Download the parquet file and load it into DuckDB."""
    if not os.path.exists(PARQUET_FILE):
        print(f"Downloading {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, PARQUET_FILE)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS trips AS
        SELECT * FROM '{PARQUET_FILE}'
    """)
    count = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    print(f"Loaded {count} rows")
    return count

class SQLTools:
    """SQL execution tools for querying the trips database."""
    
    def __init__(self, connection=None):
        """Initialize with a DuckDB connection."""
        self.con = connection or con
    
    def get_schema(self):
        """Get schema information for the trips table."""
        result = self.con.execute("DESCRIBE trips").fetchall()
        schema_info = []
        for row in result:
            col_name, col_type = row[0], row[1]
            schema_info.append(f"{col_name}: {col_type}")
        return "\n".join(schema_info)
    
    def run_sql(self, query):
        """Execute a SQL query and return results as formatted text."""
        result = self.con.execute(query).fetchall()
        columns = [desc[0] for desc in self.con.description]
        
        # Limit to 50 rows
        result = result[:50]
        
        # Format header
        lines = [" | ".join(columns)]
        lines.append("-" * len(lines[0]))
        
        # Format data rows
        for row in result:
            lines.append(" | ".join(str(val) for val in row))
        
        return "\n".join(lines)


from sql_tools import setup_database
count = setup_database()