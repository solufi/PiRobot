#!/usr/bin/env bash
# Run on the robot with sudo after copying this checkout to it.
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'Run this script with sudo.' >&2; exit 1; }
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT=/home/solufi
BACKUP="/var/backups/picar/$(date +%Y%m%d-%H%M%S)"
APP_FILES=(picar_server.py gpt_brain.py face_identity.py task_policy.py)
UNITS=(picar.service picar-update.service picar-update.path)
UPDATE_REF="${1:-main}"
[[ "$UPDATE_REF" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]{0,100}$ ]] || exit 2

"${ROOT}/picar-env/bin/python" -m py_compile "${SOURCE}/server/"*.py
bash -n "${SOURCE}/scripts/update.sh"
mkdir -p "$BACKUP/app" "$BACKUP/systemd"
chmod 700 "$BACKUP"
for file in "${APP_FILES[@]}"; do
  [[ ! -f "${ROOT}/${file}" ]] || cp -a "${ROOT}/${file}" "$BACKUP/app/"
done
for unit in "${UNITS[@]}"; do
  [[ ! -f "/etc/systemd/system/${unit}" ]] || cp -a "/etc/systemd/system/${unit}" "$BACKUP/systemd/"
done
[[ ! -f /etc/picar.env ]] || cp -a /etc/picar.env "$BACKUP/"
[[ ! -f /usr/local/sbin/picar-update ]] || cp -a /usr/local/sbin/picar-update "$BACKUP/"
[[ ! -f /etc/picar-update.env ]] || cp -a /etc/picar-update.env "$BACKUP/"
[[ ! -f /opt/picar-x/picar-x.conf ]] || cp -a /opt/picar-x/picar-x.conf "$BACKUP/"

rollback() {
  trap - ERR
  echo "Installation failed; restoring $BACKUP" >&2
  systemctl stop picar.service || true
  for file in "${APP_FILES[@]}"; do
    if [[ -f "$BACKUP/app/$file" ]]; then
      cp -a "$BACKUP/app/$file" "$ROOT/$file"
    else
      rm -f "$ROOT/$file"
    fi
  done
  if [[ -f "$BACKUP/picar-update" ]]; then
    cp -a "$BACKUP/picar-update" /usr/local/sbin/picar-update
  else
    rm -f /usr/local/sbin/picar-update
  fi
  if [[ -f "$BACKUP/picar-update.env" ]]; then
    cp -a "$BACKUP/picar-update.env" /etc/picar-update.env
  else
    rm -f /etc/picar-update.env
  fi
  systemctl disable --now picar-update.path || true
  for unit in "${UNITS[@]}"; do
    if [[ -f "$BACKUP/systemd/$unit" ]]; then
      cp -a "$BACKUP/systemd/$unit" "/etc/systemd/system/$unit"
    else
      rm -f "/etc/systemd/system/$unit"
    fi
  done
  systemctl daemon-reload
  systemctl restart picar.service || true
  exit 1
}
trap rollback ERR

systemctl stop picar.service
for file in "${APP_FILES[@]}"; do
  install -o solufi -g solufi -m 0644 "$SOURCE/server/$file" "$ROOT/$file"
done
for unit in "${UNITS[@]}"; do
  install -m 0644 "$SOURCE/systemd/$unit" "/etc/systemd/system/$unit"
done
install -o root -g root -m 0755 "$SOURCE/scripts/update.sh" /usr/local/sbin/picar-update
printf 'PICAR_UPDATE_REF=%s\n' "$UPDATE_REF" >/etc/picar-update.env
chmod 600 /etc/picar-update.env
systemctl daemon-reload
systemd-analyze verify /etc/systemd/system/picar.service /etc/systemd/system/picar-update.service /etc/systemd/system/picar-update.path
systemctl enable --now picar-update.path
systemctl restart picar.service
healthy=false
for _ in {1..30}; do
  if systemctl is-active --quiet picar.service && curl -fsS --max-time 2 http://127.0.0.1:5000/healthz >/dev/null; then
    healthy=true
    break
  fi
  sleep 1
done
[[ "$healthy" == true ]]
trap - ERR
echo "Installation OK. Backup: $BACKUP"
echo 'Interface: http://192.168.2.180:5000/'
