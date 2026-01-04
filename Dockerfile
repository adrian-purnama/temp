FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path and ensure unbuffered output for Docker logging
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Ensure stdout/stderr are not buffered
ENV PYTHONIOENCODING=utf-8

# Run the live trading bot with unbuffered output for Docker logging
# Using exec form to ensure proper signal handling
CMD ["python", "-u", "live_trade.py"]





