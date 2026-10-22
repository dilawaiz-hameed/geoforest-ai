#!/usr/bin/env bash
# Start the API, wait until it answers, then start the interface.
#
# One container runs both so the deployed demo is self-contained: the
# interface talks to the API over loopback, exactly as it does locally.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
PORT="${PORT:-7860}"

uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT}" --log-level info &
api_pid=$!

# Stop the API if the interface exits, so the container does not linger half-up.
trap 'kill "${api_pid}" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
        echo "api ready on ${API_PORT}"
        break
    fi
    if ! kill -0 "${api_pid}" 2>/dev/null; then
        echo "api exited before becoming ready" >&2
        exit 1
    fi
    sleep 1
done

# XSRF and CORS are off because the platform's reverse proxy drops the
# token Streamlit expects on /_stcore/upload_file, and file upload then
# fails with no error shown. Streamlit ignores enableXsrfProtection unless
# enableCORS is also false, so both go together. The app holds no session
# state and no credentials, so there is nothing for a forged request to do
# beyond uploading an image to itself.
exec streamlit run app/streamlit_app.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
