# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install FFmpeg package
RUN apt-get update && apt-get install -y ffmpeg

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Create a volume for the env file (mounted at runtime)
VOLUME ["/env"]

# Set environment variables from mounted .env file
ENV PYTHONPATH=/app

# Run main.py when container starts, using mounted env file
CMD ["python", "main.py"]