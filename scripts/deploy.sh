#!/usr/bin/env bash
# Deploy local changes to the PiCar-X over SSH.
# Usage: ./scripts/deploy.sh [host]   (default: solufi@192.168.2.181)
set -euo pipefail

HOST="${1:-solufi@192.168.2.181}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_VOLUME_BOOST="${PICAR_VOLUME_BOOST:-180}"
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

echo "==> Deploying to ${HOST}"

# 1) App
scp "${ROOT}/server/picar_server.py" "${HOST}:/home/solufi/picar_server.py"
scp "${ROOT}/server/gpt_brain.py"    "${HOST}:/home/solufi/gpt_brain.py"
scp "${ROOT}/server/face_identity.py" "${HOST}:/home/solufi/face_identity.py"

# 2) Systemd unit (needs sudo on the Pi)
scp "${ROOT}/systemd/picar.service" "${HOST}:/tmp/picar.service"
ssh "${HOST}" '
  # Ensure mpg123 is installed for MP3 TTS playback
  command -v mpg123 >/dev/null 2>&1 || sudo apt-get install -y -q mpg123
  sudo install -m 0644 /tmp/picar.service /etc/systemd/system/picar.service
  rm -f /tmp/picar.service
  # Clean obsolete drop-in (now baked into the unit)
  sudo rm -rf /etc/systemd/system/picar.service.d
  # Ensure /etc/picar.env exists (empty is fine — auth disabled)
  [ -f /etc/picar.env ] || (echo "# PiCar-X env (see .env.example)" | sudo tee /etc/picar.env >/dev/null && sudo chmod 600 /etc/picar.env)
  if ! sudo grep -q "^PICAR_VOLUME_BOOST=" /etc/picar.env; then
    echo "PICAR_VOLUME_BOOST='"${DEFAULT_VOLUME_BOOST}"'" | sudo tee -a /etc/picar.env >/dev/null
  fi
  mkdir -p /home/solufi/models
  if [ ! -s /home/solufi/models/face_detection_yunet_2023mar.onnx ]; then
    curl --fail --location --silent --show-error \
      -o /home/solufi/models/face_detection_yunet_2023mar.onnx \
      '"${YUNET_URL}"'
  fi
  if [ ! -s /home/solufi/models/face_recognition_sface_2021dec.onnx ]; then
    curl --fail --location --silent --show-error \
      -o /home/solufi/models/face_recognition_sface_2021dec.onnx \
      '"${SFACE_URL}"'
  fi
  touch /home/solufi/models/face_profiles.json
  chmod 600 /home/solufi/models/face_profiles.json
  sudo systemctl daemon-reload
  sudo systemctl reset-failed picar.service || true
  sudo systemctl enable picar.service
  sudo systemctl restart picar.service
  sleep 4
  systemctl is-active picar.service
  curl --fail --silent --show-error --max-time 5 http://127.0.0.1:5000/healthz >/dev/null
  ss -tlnp 2>/dev/null | grep -E ":(5000|443|80) " || true
'
echo "==> Done. Tail logs:  ssh ${HOST} journalctl -u picar.service -f"
