# Python 3.10 slim image — lightweight but complete
FROM python:3.10-slim

# System dependencies for torch and transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# So if requirements don't change, pip install won't re-run
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py .
COPY model/ ./model/

# Railway and other platforms set PORT env variable
# Default to 8000 if not set
ENV PORT=8000

# Tell Docker this container listens on this port
EXPOSE ${PORT}

# Start the FastAPI app with uvicorn
# 0.0.0.0 = accessible from outside the container
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
