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

-- Setup schema and table for myapp_db
\connect myapp_db

GRANT ALL ON SCHEMA public TO myapp_user;
GRANT USAGE, SELECT ON SEQUENCE predictions_history_id_seq TO myapp_user;
ALTER SCHEMA public OWNER TO myapp_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO myapp_user;


-- Our users table. Here we now if the user has churned
CREATE TABLE users (
  customer_id TEXT PRIMARY KEY,
  churn BOOLEAN NOT NULL
);

-- Create users_churn_data table to store features for 
CREATE TABLE users_churn_data (
  customer_id TEXT REFERENCES users(customer_id),
  account_age INTEGER,
  monthly_charges FLOAT,
  total_charges FLOAT,
  subscription_type TEXT,
  payment_method TEXT,
  paperless_billing BOOLEAN,
  content_type TEXT,
  multi_device_access BOOLEAN,
  device_registered TEXT,
  viewing_hours_per_week FLOAT,
  average_viewing_duration FLOAT,
  content_downloads_per_month INTEGER,
  genre_preference TEXT,
  user_rating FLOAT,
  support_tickets_per_month INTEGER,
  gender TEXT,
  watchlist_size INTEGER,
  parental_control BOOLEAN,
  subtitles_enabled BOOLEAN,
  PRIMARY KEY (customer_id)
);

-- To store historical predictions per user
CREATE TABLE predictions_history (
  id SERIAL PRIMARY KEY,
  customer_id TEXT REFERENCES users(customer_id),
  churn BOOLEAN,
  churn_probability FLOAT,
  predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
