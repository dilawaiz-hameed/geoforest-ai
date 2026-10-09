# GeoForest AI — API + operator interface in one image.
#
# The model layer is a build switch, not a code change. Day 1 deploys with
# INSTALL_ML=false so the CD path is proven before torch is in the picture;
# flipping it to true installs CPU torch and DeepForest and the same image
# serves real detections.
#
#   docker build -t geoforest .
#   docker build --build-arg INSTALL_ML=true -t geoforest:ml .

FROM python:3.11-slim

ARG INSTALL_ML=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    API_PORT=8000 \
    HOME=/home/app

# libgl/libglib are OpenCV's runtime dependencies, which DeepForest pulls in.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_ML" = "true" ]; then \
         pip install --no-cache-dir \
           --index-url https://download.pytorch.org/whl/cpu \
           torch torchvision \
         && pip install --no-cache-dir -r requirements-ml.txt ; \
       fi

COPY pyproject.toml ./
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY start.sh ./
RUN chmod +x start.sh && chown -R app:app /app /home/app

ENV PYTHONPATH=/app/src:/app \
    GEOFOREST_API_URL=http://127.0.0.1:8000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MPLCONFIGDIR=/home/app/.cache/matplotlib \
    TORCH_HOME=/home/app/.cache/torch

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${API_PORT}/health" || exit 1

CMD ["./start.sh"]
