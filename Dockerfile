# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies (ffmpeg is crucial)
RUN apt-get update && apt-get install -y 
    ffmpeg 
    libsndfile1 
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for moviepy to use system ffmpeg
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

# Streamlit port
ENV PORT=8080

# Run streamlit
CMD streamlit run app_local.py --server.port=${PORT} --server.address=0.0.0.0
