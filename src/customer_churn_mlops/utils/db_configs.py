from prefect_sqlalchemy import SqlAlchemyConnector
import os
import re

def get_or_create_sqlalchemy_block():
    block_name = "myapp-db"
    try:
        SqlAlchemyConnector.load(block_name)
    except ValueError:
        print(f"Block '{block_name}' not found. Creating it...")
        connector = SqlAlchemyConnector(
            sqlalchemy_url=os.getenv("MYAPP_DB_URL", "postgresql+psycopg2://myapp_user:myapppass@localhost:5432/myapp_db")
        )
        connector.save(name=block_name, overwrite=True)

def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
