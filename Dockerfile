# Dockerfile for FastAPI Recommendation API
FROM tensorflow/tensorflow:2.15.0

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Copy requirements and install
COPY app.py requirements.txt /app/
COPY data/processed/products /app/data/processed/products
COPY model /app/model
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Command to run the API
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
