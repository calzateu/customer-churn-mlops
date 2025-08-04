-- initdb/init-multi-db.sql

-- For MLflow
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH PASSWORD 'mlflowpass';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

-- For myapp
CREATE DATABASE myapp_db;
CREATE USER myapp_user WITH PASSWORD 'myapppass';
GRANT ALL PRIVILEGES ON DATABASE myapp_db TO myapp_user;

-- Grant mlflow_user all privileges on mlflow_db
\connect mlflow_db

GRANT ALL ON SCHEMA public TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow_user;
