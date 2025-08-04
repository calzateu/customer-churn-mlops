## 🧠 Problem Overview

**Customer churn** occurs when users stop using a product or service. For subscription-based businesses, predicting churn is crucial to reduce revenue loss and improve customer retention.

This project solves the **churn prediction** problem using machine learning, supported by a complete MLOps pipeline.

---

## 🚀 Solution Summary

This end-to-end project includes:

- **MLflow** for experiment tracking and model registry  
- **Prefect** for orchestrating key workflows:
  - Loading raw CSVs into **PostgreSQL**
  - **Training**, **prediction**, and **monitoring** flows

Everything is modular and production-ready, making it easy to manage, scale, and monitor over time.

---

## 📊 Dataset

Based on [this Kaggle dataset](https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset/data), it includes features like:

- `AccountAge`, `MonthlyCharges`, `TotalCharges`
- User behavior: `ViewingHoursPerWeek`, `UserRating`, `WatchlistSize`
- Preferences: `GenrePreference`, `ContentType`, `SubscriptionType`
- Target: `Churn` (1 = churned, 0 = active)


## 🛠️ Installation

1. **Set Python version (using pyenv):**

Make sure you have `pyenv` installed. The required Python version is specified in `.python-version`.

```bash
pyenv install
pyenv local $(cat .python-version)
```

2. **Install dependencies with Poetry:**

```bash
poetry install
```

> Ensure you have Poetry installed: https://python-poetry.org/docs/#installation

---

## 🚀 Usage

### 🧱 Start all services (PostgreSQL, MLflow, Prefect, Grafana, etc.)

Use the `Makefile` commands:

```bash
make up             # Build and start all services
make down           # Stop all services
make down-volumes   # Stop and remove volumes and folders
make restart        # Rebuild and restart
make logs           # View logs
make ps             # Check container status
make clean          # Full cleanup (volumes, images, networks, folders)
```

Artifacts from MLflow are stored in `./mlartifacts`.

---

### ▶️ Run a specific flow manually

You can run any Prefect flow script using:

```bash
poetry run python src/customer_churn_mlops/flows/<desired_flow>.py
```

Replace `<desired_flow>` with the actual flow filename, e.g.:

```bash
poetry run python src/customer_churn_mlops/flows/model_training_flow.py
```

---


