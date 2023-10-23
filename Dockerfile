# Use a base image
FROM tensorflow/tensorflow:2.14.0

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the source code into the container
COPY . .

# Expose necessary ports (Assuming ports 8000 and 8001 as an example)
EXPOSE 8000 8001

# Command to run the application
CMD ["sh", "run.sh"]
