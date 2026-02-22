# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies (ffmpeg is crucial)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for moviepy to use system ffmpeg
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
ENV PORT=8080

# Run app.py using gunicorn
# timeout 0 is important to allow long request processing (Cloud Run handles the actual timeout)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]