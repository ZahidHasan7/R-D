# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies (ffmpeg is required for audio)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r agent/requirements.txt
RUN pip install --no-cache-dir sentencepiece protobuf

# Expose the port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "agent.server:app", "--host", "0.0.0.0", "--port", "7860"]
