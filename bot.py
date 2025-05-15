import os
import logging
import uuid
import subprocess
import threading
import queue
import time
import glob
from pathlib import Path
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from dotenv import load_dotenv
import http.server
import socketserver

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = 'models/network-snapshot-001000.pkl'
OUTPUT_DIR = 'output'
STYLEGAN3_DIR = 'stylegan3'
CPU_GENERATOR = 'generate_cpu.py'

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Check if StyleGAN3 is set up correctly
def check_stylegan3_setup():
    """Check if StyleGAN3 is set up correctly"""
    try:
        # Check if the model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
        
        # Check if CPU generator exists
        if not os.path.exists(CPU_GENERATOR):
            logger.error(f"CPU generator not found: {CPU_GENERATOR}")
            return False
        
        # Check if stylegan3 directory exists
        if not os.path.exists(STYLEGAN3_DIR):
            logger.info(f"StyleGAN3 directory not found. Will be cloned by the generator script.")
        
        logger.info("Setup validated successfully")
        return True
    except Exception as e:
        logger.exception(f"Error checking setup: {e}")
        return False

def find_generated_file(seed):
    """Find the generated file for a given seed using different patterns"""
    patterns = [
        f"{OUTPUT_DIR}/seed{seed:04d}.png",    # Our generator's naming convention
        f"{OUTPUT_DIR}/image_{seed}.png",      # Alternative naming
        f"{OUTPUT_DIR}/{seed}.png",            # Simple seed number
        f"{OUTPUT_DIR}/*{seed}*.png"           # Any file containing the seed
    ]
    
    for pattern in patterns:
        if '*' in pattern:
            # For wildcard patterns, use glob
            files = glob.glob(pattern)
            if files:
                logger.info(f"Found generated file with wildcard pattern: {pattern} → {files[0]}")
                return files[0]
        else:
            # For exact patterns, check directly
            if os.path.exists(pattern):
                logger.info(f"Found generated file with exact pattern: {pattern}")
                return pattern
    
    # If no file is found, log all files in the output directory
    all_files = glob.glob(f"{OUTPUT_DIR}/*.png")
    logger.info(f"All files in {OUTPUT_DIR}: {all_files}")
    return None

# Operation abstraction for handling image generation requests
class Operation:
    def __init__(self, update, context, seed=None):
        self.update = update
        self.context = context
        self.seed = seed or int(uuid.uuid4().int % 100000)
        self.chat_id = update.effective_chat.id
        self.user_id = update.effective_user.id
        self.username = update.effective_user.username or str(self.user_id)
        self.timestamp = time.time()
        self.status = "pending"  # pending, processing, completed, failed
        self.image_path = None
        
    def process(self):
        """Process the image generation operation"""
        self.status = "processing"
        logger.info(f"Processing operation for user {self.username} with seed {self.seed}")
        
        # List files before generation
        before_files = set(glob.glob(f"{OUTPUT_DIR}/*.png"))
        logger.info(f"Files before generation: {before_files}")
        
        try:
            # Set CUDA_VISIBLE_DEVICES to empty to force CPU usage
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            
            # Run our standalone CPU generator
            cmd = [
                "python", CPU_GENERATOR,
                "--network", MODEL_PATH,
                "--seeds", str(self.seed),
                "--outdir", OUTPUT_DIR,
                "--trunc", "0.7"
            ]
            
            # Inform the user that CPU generation will take some time
            self.update.message.reply_text("Generating image on CPU. This might take several minutes...")
            
            # Run the command with detailed output capture
            process = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            # Log detailed output regardless of success or failure
            logger.info(f"Generator command: {' '.join(cmd)}")
            if process.stdout:
                logger.info(f"Generator output: {process.stdout}")
            if process.stderr:
                logger.error(f"Generator error: {process.stderr}")
            
            # List files after generation
            after_files = set(glob.glob(f"{OUTPUT_DIR}/*.png"))
            logger.info(f"Files after generation: {after_files}")
            
            # Find newly generated files
            new_files = after_files - before_files
            logger.info(f"New files: {new_files}")
            
            # Check if the process was successful
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, 
                                                   output=process.stdout, 
                                                   stderr=process.stderr)
            
            # First check for new files directly
            if new_files:
                # Use the first new file
                image_path = list(new_files)[0]
                logger.info(f"Using newly created file: {image_path}")
                with open(image_path, 'rb') as image_file:
                    self.update.message.reply_photo(photo=image_file, 
                                                  caption=f"Van Gogh style image (seed: {self.seed})")
                self.status = "completed"
                return True
            
            # If no new files, try to find by seed patterns
            image_path = find_generated_file(self.seed)
            if image_path:
                with open(image_path, 'rb') as image_file:
                    self.update.message.reply_photo(photo=image_file, 
                                                  caption=f"Van Gogh style image (seed: {self.seed})")
                self.status = "completed"
                return True
            else:
                self.update.message.reply_text(
                    "Sorry, I couldn't find the generated image. Please try again later.\n"
                    "The system administrator has been notified."
                )
                logger.error(f"Image generation process completed successfully, but no image file was found for seed {self.seed}.")
                self.status = "failed"
                return False
        except subprocess.CalledProcessError as e:
            error_message = f"Error code: {e.returncode}\n"
            if e.stderr:
                error_message += f"Error: {e.stderr}\n"
            if e.output:
                error_message += f"Output: {e.output}"
            
            logger.error(f"Generator process error: {error_message}")
            self.update.message.reply_text("Sorry, there was an error generating the image. "
                                          "The system administrator has been notified.")
            self.status = "failed"
            return False
        except Exception as e:
            logger.exception(f"Error generating image: {e}")
            self.update.message.reply_text("Sorry, something went wrong while generating the image. "
                                          "The system administrator has been notified.")
            self.status = "failed"
            return False


# Operation queue and worker thread
operation_queue = queue.Queue()
is_processing = False

def operation_worker():
    """Worker thread to process operations serially"""
    global is_processing
    
    while True:
        try:
            # Get operation from the queue
            operation = operation_queue.get()
            is_processing = True
            
            # Process the operation
            operation.process()
            
            # Mark the task as done
            operation_queue.task_done()
            is_processing = False
            
        except Exception as e:
            logger.exception(f"Error in operation worker: {e}")
            is_processing = False
        
        # Small sleep to prevent CPU hogging
        time.sleep(0.1)


def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text(
        'Welcome to the Van Gogh Style Image Generator Bot!\n\n'
        'Use /generate to create a new image in Van Gogh style.'
    )


def generate_image(update: Update, context: CallbackContext) -> None:
    """Queue a new image generation operation."""
    # Check if there are any pending operations
    queue_size = operation_queue.qsize()
    if queue_size > 0:
        update.message.reply_text(
            f'Your request has been queued. There are {queue_size} requests ahead of you. '
            f'Please wait...'
        )
    else:
        update.message.reply_text('Your image generation request has been received. '
                                 'Since we\'re running on CPU, this may take several minutes.')
    
    # Create a new operation and add it to the queue
    operation = Operation(update, context)
    operation_queue.put(operation)
    
    # Log the operation
    logger.info(f"Added operation to queue for user {update.effective_user.username or update.effective_user.id} " 
                f"with seed {operation.seed}")


def queue_status(update: Update, context: CallbackContext) -> None:
    """Show the current queue status."""
    queue_size = operation_queue.qsize()
    processing_status = "processing an image" if is_processing else "idle"
    
    update.message.reply_text(
        f'Queue status:\n'
        f'- Current queue size: {queue_size}\n'
        f'- Worker status: {processing_status}'
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text(
        'Bot Commands:\n'
        '/start - Start the bot\n'
        '/generate - Generate a new Van Gogh style image\n'
        '/queue - Check the current queue status\n'
        '/help - Show this help message'
    )


def error_handler(update: Update, context: CallbackContext) -> None:
    """Log the error and send a message to the user."""
    logger.error(msg=f"Exception while handling an update: {context.error}")
    if update:
        update.message.reply_text("Sorry, an error occurred. Please try again later.")


def run_health_server(port):
    """Run a simple HTTP server for health checks."""
    class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Service is running')
            
        def log_message(self, format, *args):
            # Suppress logging to avoid cluttering the output
            return
    
    with socketserver.TCPServer(("", port), HealthCheckHandler) as httpd:
        logger.info(f"Health check server running on port {port}")
        httpd.serve_forever()


def main() -> None:
    """Start the bot."""
    # Check if StyleGAN3 is set up correctly
    if not check_stylegan3_setup():
        logger.error("Setup is not correct. Exiting.")
        return
        
    # Get the Telegram token from environment variable
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("Telegram bot token not found. Please set the TELEGRAM_BOT_TOKEN environment variable.")
        return
    
    # Get health check port from environment variable or use default
    port = int(os.getenv("PORT", 8080))
    
    # Start health check server in a separate thread
    health_thread = threading.Thread(target=run_health_server, args=(port,), daemon=True)
    health_thread.start()
    
    # Start the operation worker thread
    worker_thread = threading.Thread(target=operation_worker, daemon=True)
    worker_thread.start()
    
    # Create the Updater and pass it the bot's token
    updater = Updater(token)
    
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher
    
    # Register command handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("generate", generate_image))
    dispatcher.add_handler(CommandHandler("queue", queue_status))
    
    # Register error handler
    dispatcher.add_error_handler(error_handler)
    
    # Start the Bot
    updater.start_polling()
    logger.info("Bot started")
    
    # Run the bot until the user presses Ctrl-C
    updater.idle()


if __name__ == '__main__':
    main() 