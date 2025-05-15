# Van Gogh Style Image Generator Telegram Bot

A Telegram bot that generates images in Van Gogh style using a StyleGAN3 model in CPU mode.

## Features

- Generate unique Van Gogh style images with the `/generate` command
- Serial request processing with a queue system to handle multiple users
- Containerized application with Docker and Docker Compose
- CPU-only mode for servers without GPUs
- Health check endpoint on port 8080

## Prerequisites

- Docker and Docker Compose
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- At least 4GB of RAM for the container

## Setup

1. Clone this repository:
```
git clone <your-repository-url>
cd van_gogh_gan
```

2. Create a `.env` file with your Telegram bot token:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
PORT=8080
```

## Running the Bot

Build and start the bot using Docker Compose:

```
docker-compose up -d
```

To check the logs:

```
docker-compose logs -f
```

To stop the bot:

```
docker-compose down
```

## Bot Commands

- `/start` - Start the bot and receive a welcome message
- `/generate` - Generate a new Van Gogh style image
- `/queue` - Check the current queue status
- `/help` - Display the help message with available commands

## How It Works

The bot uses a queue system to handle image generation requests:
1. When a user requests an image with `/generate`, an Operation is created and placed in the queue
2. A worker thread processes Operations one at a time (serially)
3. The user receives their image when their Operation is processed
4. If the queue has multiple Operations, users are informed of their position in the queue

## Implementation Details

This implementation uses a standalone CPU generator script (`generate_cpu.py`) that:
- Automatically sets up StyleGAN3 if not present
- Forces PyTorch to use CPU-only mode
- Implements proper error handling and logging
- Uses a consistent file naming convention

## Performance Note

This bot runs StyleGAN3 in CPU-only mode, which means image generation will be significantly slower than with GPU acceleration. Expect each image to take several minutes to generate. Users are informed about this when they request an image.

## Technical Details

- The bot uses StyleGAN3 for image generation
- Model file: `models/network-snapshot-000800.pkl`
- Generated images are stored in the `output` directory
- CPU-only version of PyTorch is used (1.9.0+cpu)
- Docker container is configured with 4-8GB of memory

## Troubleshooting

If you encounter issues with image generation:

1. Check the logs for detailed error messages:
```
docker-compose logs -f
```

2. Common issues:
   - **Memory Issues**: StyleGAN3 is memory-intensive even on CPU. The container is configured with 8GB memory limit.
   - **Model File**: Make sure the model file is correctly placed in the models directory.
   - **Permission Issues**: Ensure the bot has write permissions to the output directory.

3. If the generator fails to run, you can test it directly:
```
docker-compose exec van-gogh-bot python generate_cpu.py --network models/network-snapshot-000800.pkl --seeds 0 --outdir output --trunc 0.7
```

## Health Check

The application exposes a health check endpoint at port 8080. You can verify the service is running by visiting:

```
http://localhost:8080
``` 