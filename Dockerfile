# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install ffmpeg, a crucial system dependency for audio/video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    rubberband-cli \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Tell Cloud Run what port the app will be listening on
EXPOSE 8080

# Run the Streamlit app when the container launches
# Use the PORT environment variable provided by Cloud Run
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--server.maxUploadSize", "1024"]