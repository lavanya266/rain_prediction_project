# Use the official Python image from the Docker Hub
FROM python:3.10.12

# Add metadata with labels
LABEL maintainer="end to end mlops project"
LABEL version="1.0"
LABEL description="This is a Docker image for the rain prediction ML model with MLflow integration."

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the correct port for MLflow server (
EXPOSE 5002

# Set the MLflow tracking URI (if needed)
ENV MLFLOW_TRACKING_URI=http://172.16.51.127:5002

# Define the command to run the application or MLflow server
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./artifacts", "--host", "172.16.51.127", "--port", "5002"]
