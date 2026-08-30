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

Le script copie l'app, télécharge automatiquement les modèles YuNet/SFace s'ils manquent,
met à jour l'unit systemd, supprime l'ancien drop-in, recharge et redémarre. Il vérifie ensuite
que le service est `active` et que `/healthz` répond.
Il installe aussi Avahi et configure `robot.local` par défaut; pour un autre nom:
`PICAR_MDNS_NAME=jamal ./scripts/deploy.sh user@host`. Les clients doivent être sur le même
réseau et autoriser mDNS; sinon utiliser l'adresse IP.

## Configuration runtime

Édite `/etc/picar.env` sur le Pi (template: `.env.example`).

| Variable        | Effet                                                |
|-----------------|------------------------------------------------------|
| `PICAR_USER`    | Active basic-auth HTTP et WebSocket si défini         |
| `PICAR_PASS`    | Mot de passe basic-auth                               |
| `PICAR_PORT`    | Port d'écoute (défaut 5000)                           |
| `PICAR_CONTROL_TIMEOUT` | Arrêt automatique si la commande manuelle expire |
| `PICAR_VOLUME_BOOST` | Gain logiciel TTS (100 normal, jusqu'à 300, risque de saturation) |
| `OPENAI_API_KEY`| Phase 3 — intégration ChatGPT                         |

Recharge: `sudo systemctl restart picar.service`.

## Phases d'optimisation

- **Phase 1 — ✅** Logging, basic-auth, threaded Flask, healthz, systemd durci, drop-in fusionné
- **Phase 2** — Camera thread découplé + Flask-SocketIO pour les contrôles (latence ↓)
- **Phase 3** — Endpoint `/chat` ChatGPT (intent → actions), STT/TTS, séquences vocales courtes et interrompables

Les séquences vocales passent par `run_task` : au plus 8 étapes, 12 secondes au total et 1,5 seconde par mouvement. Une commande `stop` ou un obstacle/falaise interrompt la séquence.

## Détection YOLO optionnelle

Installer le backend sur le Raspberry Pi avec `pip install -r requirements-yolo.txt`, puis déposer un modèle nano compatible à `/home/solufi/models/yolo11n.pt` (ou définir `PICAR_YOLO_MODEL`). Avec `PICAR_PERSON_DETECTOR=auto`, YOLO est utilisé s'il est disponible; sinon HOG reste le repli automatique.

## Profils de visages locaux

La reconnaissance est optionnelle et reste locale. Installer une version OpenCV incluant
`FaceRecognizerSF`, puis déposer le modèle SFace à `/home/solufi/models/face_recognition_sface_2021dec.onnx`.
Les profils contiennent uniquement des empreintes numériques, jamais les photos. Inscrire un
profil avec une image contenant un seul visage:

```bash
curl -u "$PICAR_USER:$PICAR_PASS" -F name=Léa -F image=@lea.jpg \
  http://192.168.2.181:5000/face_profiles
curl -u "$PICAR_USER:$PICAR_PASS" http://192.168.2.181:5000/face_profiles
curl -u "$PICAR_USER:$PICAR_PASS" -X DELETE \
  -H 'Content-Type: application/json' -d '{"name":"Léa"}' \
  http://192.168.2.181:5000/face_profiles
```

Depuis l’interface, ouvrez **Paramètres**, saisissez un prénom puis cliquez sur
**ENREGISTRER PAR CAMÉRA**. Le robot prend jusqu’à sept captures et valide au moins cinq
captures stables avant d’enregistrer l’empreinte moyenne.
Le curseur **Seuil de reconnaissance** permet d’ajuster la tolérance sans SSH; une valeur
plus basse est plus stricte, une valeur plus haute accepte davantage de variations mais
augmente le risque de confusion.

Quand des profils existent, le suivi utilise uniquement un visage reconnu; sinon il refuse
le suivi. Sans modèle SFace, le robot conserve son fonctionnement précédent.

## Diagnostic rapide

```bash
ssh solufi@192.168.2.181 'systemctl status picar.service --no-pager -l'
ssh solufi@192.168.2.181 'journalctl -u picar.service -n 50 --no-pager'
curl -s http://192.168.2.181:5000/healthz
```
