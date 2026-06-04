FROM python:3.12-slim

LABEL maintainer="marcos-pereira"
LABEL description="RRT Motion Planner with pyglet GUI visualization"

# Install system dependencies for pyglet (X11 forwarding + OpenGL + FreeType fonts) and opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-dri \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxcursor1 \
    libxinerama1 \
    libxfixes3 \
    libsm6 \
    libglib2.0-0 \
    xauth \
    libfreetype6 \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies (baked into the image; code is mounted at runtime)
COPY python-scripts/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py", "smile.png", "15", "10", "50000", "40", "40", "700", "550"]
