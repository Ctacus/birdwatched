# Birdwatched

A Python-based bird watching application that detects movement in video streams and sends alerts via Telegram.

## Features

- Real-time detection of movement using OpenCV
- Sends alerts through Telegram bot
- Supports RTSP camera streaming
- Configurable alert sounds and cooldown periods
- Video recording on detection events

## Installation

### Prerequisites

Make sure you have Python 3.13 installed on your system.

### Step-by-step Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd birdwatched
   ```

2. **Create a virtual environment and activate it**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows, or source .venv/bin/activate on Unix
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Copy and configure environment variables**
    - Copy `example.env` to `.env`
    - Edit `.env` with your configuration settings

## Configuration

### Environment Variables (.env)

The application uses a `.env` file for configuration, which includes:

- `CAMERA_SOURCE`: The source of the camera (0 or RTSP URL)
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `TELEGRAM_CHAT_ID`: Chat ID where alerts are sent
- `ALERT_SOUND_PATH`: Path to alert sound file
- `MIN_CONTOUR_AREA`: Minimum area for movement detection
- `DETECTION_FRAMES_REQUIRED`: Frames required to trigger detection
- `MOVEMENT_LEVEL_REQUIRED`: Movement threshold level
- `COOLDOWN_SECONDS`: Seconds to wait before next alert
- `CLIP_SECONDS`: Duration of video clip on alert
- `FPS`: Frames per second (default: 30)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `TELEGRAM_RTMP_STREAM_KEY` and `TELEGRAM_RTMP_SERVER_URL`: For streaming via Telegram
- `RTSP_URL`: RTSP camera URL if not using default webcam

Ensure that `.env` file is present in the project root directory with valid values filled in.

## Usage

Run the main application:
```bash 
python main.py
```

The application will start monitoring your configured camera source and send alerts to Telegram when movement is detected.