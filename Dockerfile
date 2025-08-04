FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app/src"
ENV PREFECT_API_URL=${PREFECT_API_URL:-http://host.docker.internal:4200/api}

# For compiling C extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip install poetry

# Copy and install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-root

# Copy the rest of the application
COPY . .

CMD ["poetry", "run", "python", "src/customer_churn_mlops/flows/launch_flows.py"]
