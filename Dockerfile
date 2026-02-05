FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies for audio processing and Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install 3.12

ADD . /app

WORKDIR /app

RUN uv sync --locked

CMD ["uv", "run", "main.py"]