# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY backend/ ./backend/
COPY templates/ ./templates/
COPY static/ ./static/

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Set Flask environment variables
ENV FLASK_APP=app.py \
    FLASK_ENV=production

# Run the application with gunicorn for production
# If gunicorn is not in requirements.txt, we'll use Flask's built-in server
CMD ["python", "app.py"]
