FROM python:3.11-slim

LABEL authors="Lothilius"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install MCP SDK and dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy MPC server code
COPY . /app

# Expose MPC server port
EXPOSE 8084

# Run the MPC server
CMD ["python", "server.py"]
