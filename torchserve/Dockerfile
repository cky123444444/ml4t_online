# TorchServe Dockerfile
FROM pytorch/torchserve:latest

# Install curl and wget for health checks
USER root
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*
USER model-server

# Set working directory
WORKDIR /home/model-server

# Copy model files and configuration
COPY model_files/ ./model_files/
COPY raw_models/ ./raw_models/
COPY config.properties ./config.properties

# Copy scripts for model archiving
COPY deploy_model.sh ./deploy_model.sh
COPY start_torchserve.sh ./start_torchserve.sh

# Make scripts executable (as root)
USER root
RUN chmod +x deploy_model.sh start_torchserve.sh
USER model-server

# Create model store directory
RUN mkdir -p model_store

# Archive the model during build
RUN ./deploy_model.sh

# Expose TorchServe ports
EXPOSE 8080 8081 8082 7070 7071

# Health check using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/ping || exit 1

# Start TorchServe
CMD ["torchserve", "--start", "--ncs", "--model-store", "model_store", "--models", "ple.mar", "--ts-config", "config.properties"]