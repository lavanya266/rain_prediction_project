# Use a lighter TensorFlow image
FROM tensorflow/tensorflow:2.12.0-lite

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH"

# Install system packages
RUN apt-get update && apt-get install -y \
    wget \
    python3-pip \
    python3-venv \
    apt-utils \
    graphviz \
    && pip install pydot \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python3 -m venv venv \
    && venv/bin/pip install --upgrade pip \
    && venv/bin/pip install -r requirements.txt

# Copy application code
COPY . .

# Expose the MLflow tracking server port
EXPOSE 5002

# Run the application
CMD ["python", "newmlops.py"]
