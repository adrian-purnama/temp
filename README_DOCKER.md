# Docker Setup for Live Trading Bot

This guide explains how to run the live trading bot using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

2. **Run in detached mode (background):**
   ```bash
   docker-compose up -d --build
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t live-trading-bot .
   ```

2. **Run the container:**
   ```bash
   docker run -it --name live-trading-bot \
     -v $(pwd)/logs:/app/logs \
     -v $(pwd)/data:/app/data \
     live-trading-bot
   ```

3. **Run in detached mode:**
   ```bash
   docker run -d --name live-trading-bot \
     -v $(pwd)/logs:/app/logs \
     -v $(pwd)/data:/app/data \
     live-trading-bot
   ```

4. **View logs:**
   ```bash
   docker logs -f live-trading-bot
   ```

5. **Stop the container:**
   ```bash
   docker stop live-trading-bot
   docker rm live-trading-bot
   ```

## Configuration

Since API keys are hardcoded in `trader/live/config.py`, you don't need to set environment variables. However, you can override settings using command-line arguments:

```bash
docker run -it live-trading-bot python live_trading.py --symbol ETHUSDT --timeframe 1h
```

## Volumes

The Docker setup mounts two directories:

- `./logs` → `/app/logs` - For log files
- `./data` → `/app/data` - For CSV trade logs and position backups

Make sure these directories exist or Docker will create them automatically.

## Viewing the Dashboard

The terminal dashboard will be visible in the Docker logs. To see it in real-time:

```bash
docker logs -f live-trading-bot
```

## Stopping the Bot

To stop the bot gracefully:

1. **Using Docker Compose:**
   ```bash
   docker-compose down
   ```

2. **Using Docker:**
   ```bash
   docker stop live-trading-bot
   ```

3. **Using kill switch file:**
   Create a file named `kill_switch.txt` in the data directory:
   ```bash
   touch data/kill_switch.txt
   ```

## Troubleshooting

### Container exits immediately

Check the logs:
```bash
docker logs live-trading-bot
```

### Permission issues with volumes

On Linux, you may need to adjust permissions:
```bash
sudo chown -R $USER:$USER logs data
```

### Rebuild after code changes

```bash
docker-compose build --no-cache
docker-compose up
```

## Notes

- The container runs in the foreground by default to show the dashboard
- Use `-d` flag for detached mode if you want it to run in the background
- Logs are persisted in the `./logs` directory
- Trade data and position backups are saved in `./data` directory

