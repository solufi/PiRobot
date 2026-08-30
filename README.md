# PiCar-X — Serveur web optimisé

Contrôleur web pour SunFounder PiCar-X (Raspberry Pi, Debian 13, Python 3.13).
Source de vérité versionnée pour le code qui tourne sur `solufi@192.168.2.181`.

## Layout

```
server/picar_server.py        # App Flask (vidéo MJPEG + contrôles + face tracking)
systemd/picar.service         # Unit systemd durcie (déployée vers /etc/systemd/system/)
scripts/deploy.sh             # scp + restart + healthcheck
.env.example                  # Variables d'environnement -> /etc/picar.env sur le Pi
```

## Déploiement

```bash
./scripts/deploy.sh                 # défaut: solufi@192.168.2.181
./scripts/deploy.sh user@host       # autre cible
```

Le script copie l'app, met à jour l'unit systemd, supprime l'ancien drop-in, recharge et redémarre. Il échoue immédiatement si le service ne devient pas `active`.

## Configuration runtime

Édite `/etc/picar.env` sur le Pi (template: `.env.example`).

| Variable        | Effet                                                |
|-----------------|------------------------------------------------------|
| `PICAR_USER`    | Active basic-auth HTTP si défini                      |
| `PICAR_PASS`    | Mot de passe basic-auth                               |
| `PICAR_PORT`    | Port d'écoute (défaut 5000)                           |
| `OPENAI_API_KEY`| Phase 3 — intégration ChatGPT                         |

Recharge: `sudo systemctl restart picar.service`.

## Phases d'optimisation

- **Phase 1 — ✅** Logging, basic-auth, threaded Flask, healthz, systemd durci, drop-in fusionné
- **Phase 2** — Camera thread découplé + Flask-SocketIO pour les contrôles (latence ↓)
- **Phase 3** — Endpoint `/chat` ChatGPT (intent → actions), STT/TTS

## Diagnostic rapide

```bash
ssh solufi@192.168.2.181 'systemctl status picar.service --no-pager -l'
ssh solufi@192.168.2.181 'journalctl -u picar.service -n 50 --no-pager'
curl -s http://192.168.2.181:5000/healthz
```
