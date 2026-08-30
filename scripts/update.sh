#!/usr/bin/env bash
set -euo pipefail

REF="${1:-main}"
ROOT="/home/solufi"
STATUS="${ROOT}/update-status.json"
LOCK="${ROOT}/picar-update.lock"
ARCHIVE_URL="https://github.com/solufi/PiRobot/archive/refs/heads/${REF}.tar.gz"

mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || exit 0

write_status() {
  local state="$1"
  local message="$2"
  printf '{"state":"%s","message":"%s","ref":"%s"}\n' \
    "$state" "$message" "$REF" >"${STATUS}.tmp"
  mv "${STATUS}.tmp" "$STATUS"
}

tmp="$(mktemp -d)"
backup="${ROOT}/update-backup-$(date +%s)"
trap 'rm -rf "$tmp"' EXIT

write_status "downloading" "Téléchargement de la version ${REF}"
curl --fail --location --silent --show-error --max-time 120 \
  -o "${tmp}/source.tar.gz" "$ARCHIVE_URL"
tar -xzf "${tmp}/source.tar.gz" -C "$tmp"
source_dir="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)"

for file in server/picar_server.py server/gpt_brain.py server/face_identity.py server/task_policy.py; do
  test -s "${source_dir}/${file}"
done
python3 -m py_compile \
  "${source_dir}/server/picar_server.py" \
  "${source_dir}/server/gpt_brain.py" \
  "${source_dir}/server/face_identity.py" \
  "${source_dir}/server/task_policy.py"

write_status "installing" "Installation de la version ${REF}"
mkdir "$backup"
for file in picar_server.py gpt_brain.py face_identity.py task_policy.py; do
  cp "${ROOT}/${file}" "${backup}/${file}"
done
cp "${source_dir}/server/"{picar_server.py,gpt_brain.py,face_identity.py,task_policy.py} "$ROOT/"

pkill -TERM -f '[p]icar_server.py' || true
healthy=false
for _ in $(seq 1 30); do
  if curl --fail --silent --show-error --max-time 2 \
      http://127.0.0.1:5000/healthz >/dev/null; then
    healthy=true
    break
  fi
  sleep 1
done

if [ "$healthy" != true ]; then
  write_status "rollback" "Échec du redémarrage, restauration de la version précédente"
  cp "${backup}/"{picar_server.py,gpt_brain.py,face_identity.py,task_policy.py} "$ROOT/"
  pkill -TERM -f '[p]icar_server.py' || true
  write_status "error" "Mise à jour annulée et version précédente restaurée"
  exit 1
fi

rm -rf "$backup"
write_status "success" "Mise à jour installée; le service répond correctement"
