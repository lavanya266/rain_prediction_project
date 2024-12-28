# Base image with Python and TensorFlow
FROM tensorflow/tensorflow:2.12.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install required system packages
RUN apt-get update && apt-get install -y \
    wget \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code to the container
COPY . /app

# Expose the MLflow tracking server port
EXPOSE 5002

# Run the application
CMD ["python", "newmlops.py"]
