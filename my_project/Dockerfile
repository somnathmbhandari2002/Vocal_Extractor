FROM python:3.10-slim

# Install system dependencies (FFmpeg + git for Demucs)
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy all your project files into the image
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port used by FastAPI (matches PORT=10000 on Render)
EXPOSE 10000

# Start the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
