# PiCar-X — Serveur web optimisé

Contrôleur web pour SunFounder PiCar-X (Raspberry Pi, Debian 13, Python 3.13).
Source de vérité versionnée pour le code qui tourne sur `solufi@192.168.2.181`.

L'interface écoute sur le **port 5000** : `http://192.168.2.180:5000/`
sur Ethernet lors du diagnostic du 21 octobre 2026. Le Pi possède aussi une
interface Wi-Fi en `.181`; les adresses DHCP peuvent changer. Utiliser l'adresse
de l'interface accessible depuis le poste client, avec `:5000`.

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
met à jour les units systemd, conserve les overrides locaux, recharge et redémarre. Il vérifie ensuite
que le service est `active` et que `/healthz` répond.
Il installe aussi Avahi et configure `robot.local` par défaut; pour un autre nom:
`PICAR_MDNS_NAME=jamal ./scripts/deploy.sh user@host`. Les clients doivent être sur le même
réseau et autoriser mDNS; sinon utiliser l'adresse IP.
Après ce premier déploiement, le bouton **METTRE À JOUR** sur l’écran principal télécharge la
branche `main` depuis GitHub, vérifie la syntaxe, remplace uniquement les fichiers applicatifs
et redémarre automatiquement le service. En cas d’échec du healthcheck, la version précédente
est restaurée; les modèles, profils et `/etc/picar.env` ne sont pas modifiés.
L’interface affiche la progression, attend la reconnexion après redémarrage puis
se recharge automatiquement. Les quatre modules Python du serveur contiennent
l’application et son interface; les unités système et le moteur de mise à jour
s’installent avec le script de déploiement.
La demande est exécutée par `picar-update.path` et `picar-update.service`, dans un
processus indépendant du contrôleur. Cela évite d'interrompre la mise à jour lorsque
le contrôleur redémarre. `PICAR_UPDATE_REF` dans `/etc/picar-update.env` permet de
suivre une branche précise (défaut `main`).

Pour installer une copie préparée sur le Pi avec sauvegarde et restauration en cas
d'échec : `sudo bash scripts/install-staged.sh [branche-de-mise-à-jour]`.
Les sauvegardes restent dans `/var/backups/picar/`. Le script conserve la clé API,
les profils et la calibration existants.

## Fiabilité du suivi autonome

Toutes les commandes de déplacement du suivi passent par le même contrôle de
sécurité que les commandes manuelles. STOP désactive le suivi avant d'arrêter les
moteurs. Un flux caméra figé, des capteurs bloqués, un obstacle ou une falaise
interrompent les mouvements concernés. Trois mesures ultrason invalides de suite
invalident l'ancienne distance : elle ne peut plus autoriser une marche avant.
Un message dans l'interface signale les capteurs indisponibles.

L'encodage JPEG est partagé entre les spectateurs; la reconnaissance ne lance plus
deux détections du même visage à chaque cycle. La cadence du suivi inclut le temps
de calcul. Sur le Pi 4 mesuré, YuNet prenait environ 35 ms à 320×240; cela ne
remplace pas une validation du suivi et des arrêts sur le robot en conditions réelles.

Tests sans matériel : `python3 -m unittest discover -s tests -v`.

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
