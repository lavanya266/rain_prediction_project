# Base image with Python and TensorFlow
FROM tensorflow/tensorflow:2.12.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required system packages
RUN apt-get update && apt-get install -y \
    wget \
    python3-pip \
    python3-venv \
    apt-utils \
    graphviz \
    && pip install pydot \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies inside virtual environment
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

