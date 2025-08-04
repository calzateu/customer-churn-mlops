import os

DB_URL = os.getenv("MYAPP_DB_URL", "postgresql+psycopg2://myapp_user:myapppass@localhost:5432/myapp_db")
