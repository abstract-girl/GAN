FROM python:3.9-slim

WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone StyleGAN3 repository
RUN git clone https://github.com/NVlabs/stylegan3.git /app/stylegan3

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install StyleGAN3 requirements
RUN pip install --no-cache-dir ninja scipy lpips click pillow numpy

# Create output directory and ensure proper permissions
RUN mkdir -p output && chmod 777 output

# Copy model and application files
COPY models/ models/
COPY bot.py .
COPY generate_cpu.py .
COPY .env .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=""

# Run the bot
CMD ["python", "bot.py"] 