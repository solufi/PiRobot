"""GPT brain for PiCar-X.

Exposes a `chat(message)` function that:
  - sends the user message to OpenAI with tool definitions for robot actions
  - executes returned tool calls via callbacks registered by the main server
  - returns a final spoken reply (text) + the list of executed actions

Also exposes:
  - transcribe(audio_path) -> str   (Whisper STT)
  - synthesize(text, out_path)      (OpenAI TTS)
"""
from __future__ import annotations
import os, json, logging, time
from typing import Callable, Any

log = logging.getLogger("picar.gpt")

# Lazy import so the server still starts if the package is missing
try:
    from openai import OpenAI
    _AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _AVAILABLE = False

MODEL_CHAT = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# gpt-4o-mini-transcribe is ~2x faster than whisper-1 for short clips
MODEL_STT = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
# tts-1 has lower time-to-first-byte than gpt-4o-mini-tts
MODEL_TTS = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "alloy")

SYSTEM_PROMPT = """Tu es le cerveau d'un petit robot-voiture nommé "Jamal" (PiCar).
Tu parles aux enfants et adultes en français, de façon courte, fun et bienveillante.

Tu peux faire DEUX choses :
1. Répondre à toute question générale (culture, math, blague, météo connue, conseils, etc.)
   en utilisant tes propres connaissances, sans appeler de tool. Sois bref (1-3 phrases).
2. Exécuter des actions physiques sur le robot — dans ce cas tu DOIS appeler le tool
   correspondant. N'invente JAMAIS l'effet : appelle vraiment le tool.

Si une demande mélange les deux (« avance et raconte-moi une blague »), appelle le tool
pour l'action puis ajoute la réponse parlée.

Mapping des intentions courantes vers les tools :
- "avance", "va devant", "tout droit"  -> drive(direction="forward", duration_ms=600)
- "recule", "va derrière"               -> drive(direction="backward", duration_ms=600)
- "tourne à gauche", "gauche"          -> drive(direction="left", duration_ms=600)
- "tourne à droite", "droite"          -> drive(direction="right", duration_ms=600)
- "stop", "arrête", "danger"           -> drive(direction="stop") ou stop_all()
- "suis-moi", "follow me", "viens"      -> set_follow_me(enabled=true)
- "arrête de me suivre"                -> set_follow_me(enabled=false)
- "regarde en haut/bas/gauche/droite"  -> set_camera(pan=..., tilt=...)
- "danse", "fais le clown"             -> dance()
- "monte/baisse le son", "plus fort"    -> set_volume(percent=...)
- "qu'as-tu devant toi", "distance"    -> get_distance()
- "regarde le sol", "es-tu au bord"    -> get_grayscale()
- "active/coupe la sécurité"           -> set_safety(enabled=...)
- "arrête d'écouter", "tais-toi"       -> set_listening(enabled=false)
- "écoute-moi", "réveille-toi"         -> set_listening(enabled=true)

Règles de sécurité STRICTES :
- Chaque drive ne dure JAMAIS plus de 1500 ms.
- Vitesse par défaut entre 30 et 80 ; au-dessus seulement si on dit "fonce", "à fond".
- En cas de doute -> stop_all().

Tu réponds TOUJOURS par une TRÈS courte phrase parlée (max 12 mots), MÊME quand tu fais une action.
Ne décris pas les outils, agis simplement. Pas de listes, pas d'emojis.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "drive",
        "description": "Fait avancer/reculer/tourner le robot pendant une courte durée puis l'arrête.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["forward", "backward", "left", "right", "stop"]},
                "duration_ms": {"type": "integer", "minimum": 100, "maximum": 1500, "default": 600},
            },
            "required": ["direction"],
        },
    }},
    {"type": "function", "function": {
        "name": "set_speed",
        "description": "Règle la vitesse de déplacement (10 lent, 50 modéré, 100 rapide).",
        "parameters": {
            "type": "object",
            "properties": {"speed": {"type": "integer", "minimum": 10, "maximum": 100}},
            "required": ["speed"],
        },
    }},
    {"type": "function", "function": {
        "name": "set_camera",
        "description": "Oriente la caméra. pan = horizontal (-60 gauche, 60 droite), tilt = vertical (-40 bas, 40 haut).",
        "parameters": {
            "type": "object",
            "properties": {
                "pan": {"type": "integer", "minimum": -60, "maximum": 60},
                "tilt": {"type": "integer", "minimum": -40, "maximum": 40},
            },
        },
    }},
    {"type": "function", "function": {
        "name": "set_follow_me",
        "description": "Active/désactive le mode 'suis-moi' (le robot suit ton visage avec la caméra et roule doucement vers toi).",
        "parameters": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
        },
    }},
    {"type": "function", "function": {
        "name": "set_volume",
        "description": "Règle le volume du haut-parleur du robot (0 muet, 100 maximum). À utiliser quand on dit 'monte le son', 'baisse le son', 'plus fort', 'moins fort', 'mute'.",
        "parameters": {
            "type": "object",
            "properties": {"percent": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["percent"],
        },
    }},
    {"type": "function", "function": {
        "name": "dance",
        "description": "Petite chorégraphie : tourne la tête à gauche/droite et frétille.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "stop_all",
        "description": "Arrêt complet immédiat (moteurs + suivi).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_distance",
        "description": "Retourne la distance en cm mesurée par le capteur ultrason avant. Utile avant d'avancer ou pour répondre 'qu'est-ce qu'il y a devant toi'.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_grayscale",
        "description": "Retourne les 3 valeurs des capteurs de niveaux de gris au sol [gauche, milieu, droite] et un flag 'cliff' (vide/falaise).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "set_safety",
        "description": "Active/désactive l'arrêt automatique de sécurité (obstacle proche ou falaise). Garde-le activé par défaut.",
        "parameters": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
        },
    }},
    {"type": "function", "function": {
        "name": "set_listening",
        "description": "Active ou désactive l'écoute continue avec le mot de réveil 'Jamal'. À utiliser quand l'utilisateur dit 'écoute-moi', 'réveille-toi', 'arrête d'écouter', 'tais-toi'.",
        "parameters": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
        },
    }},
]


class GPTBrain:
    def __init__(self, action_handlers: dict[str, Callable[..., Any]]):
        self.handlers = action_handlers
        self.client = None
        self.history: list[dict] = []  # short rolling memory
        self.max_history = 6   # smaller history -> faster prompts
        if _AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI()
            log.info("GPT brain ready (model=%s)", MODEL_CHAT)
        else:
            log.warning("GPT brain disabled (openai installed=%s, key set=%s)",
                        _AVAILABLE, bool(os.environ.get("OPENAI_API_KEY")))

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _run_tool(self, name: str, args: dict) -> dict:
        handler = self.handlers.get(name)
        if not handler:
            return {"ok": False, "error": f"unknown tool {name}"}
        try:
            result = handler(**args)
            if isinstance(result, dict):
                return {"ok": True, **result}
            return {"ok": True}
        except Exception as e:
            log.exception("tool %s failed", name)
            return {"ok": False, "error": str(e)}

    def chat(self, message: str, max_steps: int = 4) -> dict:
        if not self.enabled:
            return {"reply": "GPT n'est pas configuré sur ce robot.", "actions": [], "error": "disabled"}

        self.history.append({"role": "user", "content": message})
        self.history = self.history[-self.max_history:]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        actions: list[dict] = []

        for step in range(max_steps):
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL_CHAT, messages=msgs, tools=TOOLS, temperature=0.4,
                )
            except Exception as e:
                log.warning("OpenAI call failed: %s", e)
                err = str(e)
                if "insufficient_quota" in err or "exceeded your current quota" in err:
                    msg = "Le compte OpenAI n'a plus de crédits. Recharge ton compte sur platform.openai.com."
                elif "invalid_api_key" in err or "Incorrect API key" in err:
                    msg = "Clé OpenAI invalide."
                else:
                    msg = "Erreur OpenAI : " + err.splitlines()[0][:200]
                return {"reply": msg, "actions": actions, "error": err}
            else:
                pass
            m = resp.choices[0].message
            msgs.append(m.model_dump(exclude_none=True))

            if not m.tool_calls:
                reply = (m.content or "").strip() or "Ok."
                self.history.append({"role": "assistant", "content": reply})
                return {"reply": reply, "actions": actions}

            for call in m.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except Exception:
                    args = {}
                log.info("tool call: %s(%s)", name, args)
                result = self._run_tool(name, args)
                actions.append({"name": name, "args": args, "result": result})
                msgs.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                })

        return {"reply": "(trop d'étapes)", "actions": actions}

    def transcribe(self, audio_path: str) -> str:
        if not self.enabled:
            return ""
        with open(audio_path, "rb") as f:
            t = self.client.audio.transcriptions.create(
                model=MODEL_STT, file=f, language="fr"
            )
        return (t.text or "").strip()

    def synthesize(self, text: str, out_path: str) -> str:
        if not self.enabled:
            return ""
        with self.client.audio.speech.with_streaming_response.create(
            model=MODEL_TTS, voice=TTS_VOICE, input=text, response_format="mp3",
        ) as r:
            r.stream_to_file(out_path)
        return out_path
