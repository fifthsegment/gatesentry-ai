# Use a base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y imagemagick && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --only-binary Pillow Pillow
RUN pip3 install --no-cache-dir -r requirements.txt gunicorn 

# Copy the source code into the container
COPY . .

# Expose necessary ports (Assuming ports 8000 and 8001 as an example)
EXPOSE 8000 8001

# Command to run the application
CMD ["sh", "run.sh"]
