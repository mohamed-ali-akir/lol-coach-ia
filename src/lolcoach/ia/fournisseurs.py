"""Les modèles d'IA utilisables pour regarder les captures et rédiger le rapport.

Deux usages :
- **vision** : regarder les images d'un moment de la partie et renvoyer un JSON (erreurs du
  catalogue) ;
- **rédaction** : écrire le rapport de coaching à partir des faits.

Fournisseurs (variables LOLCOACH_VISION et LOLCOACH_REDACTEUR) :
- `gemini`      : API Gemini (pip install "lolcoach[gemini]", clé GEMINI_API_KEY) ;
- `agy`         : Gemini par la ligne de commande Google Antigravity (`agy -p`), avec ton
                  compte Google : c'est le « relais » ;
- `claude-code` : Claude Code en mode non interactif (`claude -p`), avec ton abonnement ;
- `claude`      : API Claude (rédaction seulement, clé ANTHROPIC_API_KEY) ;
- `commande`    : n'importe quelle commande, décrite par LOLCOACH_VISION_CMD ou
                  LOLCOACH_REDACTEUR_CMD (voir `CommandProvider`).

Tout ce qui revient d'une IA est relu et validé ailleurs (`suivi/store.py`) : ici, on se
contente de l'appeler et d'extraire sa réponse.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

DEFAULT_GEMINI_MODEL = "gemini-4-argon"
COMMAND_TIMEOUT_S = 600
MAX_IMAGES = 40
MAX_OUTPUT_CHARS = 200_000
# Au-delà, la consigne passe par un fichier : une ligne de commande Windows est limitée à
# environ 32 000 caractères.
LONG_PROMPT = 6_000

# Gabarits des commandes connues. Jetons remplacés : {prompt} (texte de la consigne),
# {prompt_file}, {schema_file} (schéma JSON attendu), {images} (chemins, un jeton chacun),
# {dossier} (dossier des images).
PRESETS: dict[str, dict[str, str]] = {
    "agy": {
        "vision": "agy -p {prompt} --output-format json --json-schema {schema_file}",
        "redaction": "agy -p {prompt} --output-format json",
    },
    "claude-code": {
        "vision": "claude -p {prompt} --output-format json --allowedTools Read",
        "redaction": "claude -p {prompt} --output-format json --allowedTools Read",
    },
}


class AIError(Exception):
    """Fournisseur absent, mal configuré, ou réponse inutilisable."""


class VisionProvider(Protocol):
    name: str

    def look(self, prompt: str, images: Sequence[Path], schema: dict[str, Any]
             ) -> dict[str, Any]: ...


class TextProvider(Protocol):
    name: str

    def write(self, prompt: str) -> str: ...


# --- Extraction des réponses ---

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Premier objet JSON d'une réponse : brut, dans un bloc ```json, ou dans l'enveloppe
    d'une ligne de commande (`{"result": "..."}` de Claude Code, champ texte d'agy)."""
    text = text.strip()
    candidates: list[Any] = []
    with contextlib.suppress(ValueError):
        candidates.append(json.loads(text))
    candidates += [m.group(1) for m in _FENCE_RE.finditer(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except ValueError:
                continue
        if not isinstance(candidate, dict):
            continue
        inner = _envelope_payload(candidate)
        if inner is not None:
            return inner
        return candidate
    raise AIError("la réponse ne contient pas d'objet JSON")


def _envelope_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """Contenu utile d'une enveloppe de ligne de commande, ou None si `data` n'en est pas une."""
    for key in ("structured_output", "structuredOutput", "structured_result"):
        if isinstance(data.get(key), dict):
            return dict(data[key])
    for key in ("result", "response", "text", "output"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and "{" in value:
            try:
                return extract_json(value)
            except AIError:
                return None
    return None


def extract_text(stdout: str) -> str:
    """Texte d'une réponse de ligne de commande : l'enveloppe JSON si elle existe."""
    stripped = stdout.strip()
    try:
        data = json.loads(stripped)
    except ValueError:
        return stripped
    if isinstance(data, dict):
        if data.get("is_error") or str(data.get("status", "")).upper() == "ERROR":
            raise AIError(f"la commande signale une erreur : {str(data)[:300]}")
        for key in ("result", "response", "text", "output"):
            if isinstance(data.get(key), str):
                return str(data[key]).strip()
    return stripped


# --- Ligne de commande (agy, claude -p, ou autre) ---

Runner = Callable[[list[str], float], subprocess.CompletedProcess[str]]


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace", check=False)


class CommandProvider:
    """Appelle une commande locale. Le gabarit est découpé en jetons AVANT remplacement :
    une consigne ou un chemin contenant des espaces reste un seul argument, et rien ne
    passe par un interpréteur de commandes."""

    def __init__(self, template: str, name: str = "commande", timeout: float = COMMAND_TIMEOUT_S,
                 runner: Runner | None = None) -> None:
        self.tokens = shlex.split(template)
        if not self.tokens:
            raise AIError("commande vide")
        if shutil.which(self.tokens[0]) is None and runner is None:
            raise AIError(f"commande introuvable : {self.tokens[0]} (est-elle installée et "
                          "dans le PATH ?)")
        self.name = name
        self._timeout = timeout
        self._runner = runner or _run

    def _args(self, prompt: str, images: Sequence[Path], schema: dict[str, Any] | None,
              workdir: Path) -> list[str]:
        prompt_file = workdir / "consigne.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        schema_file = workdir / "schema.json"
        schema_file.write_text(json.dumps(schema or {"type": "object"}), encoding="utf-8")
        folder = str(images[0].parent.resolve()) if images else str(workdir)
        if len(prompt) > LONG_PROMPT:
            prompt = (f"Ta consigne complète est dans le fichier {prompt_file.resolve()} : "
                      "lis-le en entier avec ton outil de lecture de fichiers, puis suis-la.")
        args: list[str] = []
        for token in self.tokens:
            if token == "{images}":
                args += [str(p.resolve()) for p in images]
                continue
            args.append(token.replace("{prompt_file}", str(prompt_file))
                        .replace("{schema_file}", str(schema_file))
                        .replace("{dossier}", folder)
                        .replace("{prompt}", prompt))
        return args

    def _call(self, prompt: str, images: Sequence[Path], schema: dict[str, Any] | None) -> str:
        with tempfile.TemporaryDirectory(prefix="lolcoach-") as tmp:
            args = self._args(prompt, images, schema, Path(tmp))
            try:
                done = self._runner(args, self._timeout)
            except subprocess.TimeoutExpired as exc:
                raise AIError(f"{self.name} n'a pas répondu en {self._timeout:.0f} s") from exc
            except OSError as exc:
                raise AIError(f"{self.name} n'a pas pu être lancé : {exc}") from exc
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "").strip().splitlines()[-1:] or [""]
            raise AIError(f"{self.name} a échoué (code {done.returncode}) : {detail[0][:300]}")
        return (done.stdout or "")[:MAX_OUTPUT_CHARS]

    def look(self, prompt: str, images: Sequence[Path], schema: dict[str, Any]
             ) -> dict[str, Any]:
        listing = "\n".join(f"- {p.resolve()}" for p in images[:MAX_IMAGES])
        full = (f"{prompt}\n\nImages à regarder (ouvre chacune avec ton outil de lecture de "
                f"fichiers, dans l'ordre) :\n{listing}\n\nRéponds uniquement par l'objet JSON.")
        return extract_json(self._call(full, images[:MAX_IMAGES], schema))

    def write(self, prompt: str) -> str:
        text = extract_text(self._call(prompt, [], None))
        if not text:
            raise AIError(f"{self.name} a renvoyé une réponse vide")
        return text


# --- API Gemini ---

class GeminiProvider:
    """API Gemini via le SDK officiel `google-genai` (clé GEMINI_API_KEY ou GOOGLE_API_KEY)."""

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, client: Any = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise AIError('SDK absent : pip install "lolcoach[gemini]"') from exc
        self._types = types
        self._client = client or genai.Client()
        self._model = model
        self._sleep = sleep
        self.name = f"gemini:{model}"

    def _generate(self, contents: list[Any], schema: dict[str, Any] | None) -> Any:
        config = None
        if schema is not None:
            config = self._types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=schema)
        try:
            return self._client.models.generate_content(model=self._model, contents=contents,
                                                        config=config)
        except Exception as exc:  # le SDK lève des types variés selon la panne
            raise AIError(f"l'API Gemini a échoué : {type(exc).__name__}: {str(exc)[:300]}"
                          ) from exc

    def look(self, prompt: str, images: Sequence[Path], schema: dict[str, Any]
             ) -> dict[str, Any]:
        parts: list[Any] = []
        for path in images[:MAX_IMAGES]:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            parts.append(self._types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
        response = self._generate([*parts, prompt], schema)
        return extract_json(getattr(response, "text", "") or "")

    def look_video(self, prompt: str, url: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Une vidéo YouTube publique, regardée directement par Gemini."""
        part = self._types.Part(file_data=self._types.FileData(file_uri=url))
        response = self._generate([part, prompt], schema)
        return extract_json(getattr(response, "text", "") or "")

    def write(self, prompt: str) -> str:
        text = getattr(self._generate([prompt], None), "text", "") or ""
        if not text.strip():
            raise AIError("Gemini a renvoyé une réponse vide")
        return text.strip()


class ClaudeTextProvider:
    """Rédaction par l'API Claude, en réutilisant le client du « mot du coach »."""

    def __init__(self, model: str | None = None) -> None:
        from lolcoach.coach.llm import DEFAULT_CLAUDE_MODEL, ClaudeProvider, LLMError

        try:
            self._provider = ClaudeProvider(model or DEFAULT_CLAUDE_MODEL, effort="medium")
        except LLMError as exc:
            raise AIError(str(exc)) from exc
        self.name = self._provider.name

    def write(self, prompt: str) -> str:
        from lolcoach.coach.llm import LLMError

        try:
            return self._provider.complete(
                "Tu es un coach League of Legends exigeant et bienveillant.", prompt).strip()
        except LLMError as exc:
            raise AIError(str(exc)) from exc


# --- Choix par variables d'environnement ---

def _command(kind: str, choice: str, env_var: str) -> CommandProvider:
    template = os.environ.get(env_var, "").strip()
    if choice in PRESETS:
        template = template or PRESETS[choice][kind]
    if not template:
        raise AIError(f"{env_var} est vide : décris la commande à lancer")
    return CommandProvider(template, name=choice)


def vision_from_env() -> VisionProvider | None:
    """LOLCOACH_VISION : aucun (défaut), gemini, agy, claude-code ou commande."""
    choice = os.environ.get("LOLCOACH_VISION", "").strip().lower()
    if choice in ("", "aucun"):
        return None
    if choice == "gemini":
        return GeminiProvider(os.environ.get("LOLCOACH_VISION_MODEL", "").strip()
                              or DEFAULT_GEMINI_MODEL)
    if choice in ("agy", "claude-code", "commande"):
        return _command("vision", choice, "LOLCOACH_VISION_CMD")
    raise AIError(f"LOLCOACH_VISION inconnu : {choice!r} (aucun, gemini, agy, claude-code, "
                  "commande)")


def writer_from_env() -> TextProvider | None:
    """LOLCOACH_REDACTEUR : aucun (défaut), gemini, agy, claude-code, claude ou commande."""
    choice = os.environ.get("LOLCOACH_REDACTEUR", "").strip().lower()
    model = os.environ.get("LOLCOACH_REDACTEUR_MODEL", "").strip()
    if choice in ("", "aucun"):
        return None
    if choice == "gemini":
        return GeminiProvider(model or DEFAULT_GEMINI_MODEL)
    if choice == "claude":
        return ClaudeTextProvider(model or None)
    if choice in ("agy", "claude-code", "commande"):
        return _command("redaction", choice, "LOLCOACH_REDACTEUR_CMD")
    raise AIError(f"LOLCOACH_REDACTEUR inconnu : {choice!r} (aucun, gemini, agy, claude-code, "
                  "claude, commande)")
