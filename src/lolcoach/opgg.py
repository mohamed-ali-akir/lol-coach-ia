"""Build du patch en cours d'après OP.GG.

Les fiches tirées de vidéos vieillissent à chaque patch ; le build (objets, runes, sorts,
ordre des compétences) vient donc d'OP.GG, par son serveur public prévu pour les applis
(« OP.GG MCP Server », https://github.com/opgginc/opgg-mcp), le même que LoL Companion :

    POST https://mcp-api.op.gg/mcp   (JSON-RPC 2.0, méthode « tools/call »,
                                     outil « lol_get_champion_analysis »)

On n'envoie QUE le champion, le rôle, le mode (classée) et la langue : jamais le Riot ID,
le puuid ni aucune donnée de partie. La réponse n'est pas du JSON mais un texte propre à
OP.GG : des lignes « class Nom: champ1,champ2 » puis une expression « Nom(valeur, …) »,
lue par `parse_opgg_text`.

Robustesse : délai de 15 s, réponse limitée à 512 Ko, forme vérifiée, et
`champion_build` ne lève jamais d'exception : en cas de problème, il renvoie
(None, raison) et le coach continue sans build. Les réponses sont gardées 24 h dans
`donnees/cache/opgg` (dossier ignoré par Git). LOLCOACH_OPGG=0 coupe l'appel.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

OPGG_URL = "https://mcp-api.op.gg/mcp"
TOOL = "lol_get_champion_analysis"
LANG = "fr_FR"  # noms des objets et des runes en français
TIMEOUT_S = 15.0
MAX_BYTES = 512_000
CACHE_DIR = Path("donnees") / "cache" / "opgg"
CACHE_TTL_S = 24 * 3600
MAX_DEPTH = 40  # imbrication maximale acceptée dans la réponse
USER_AGENT = ("lolcoach (coach League of Legends personnel ; "
              "github.com/mohamed-ali-akir/lol-coach-ia)")

FIELDS = [
    "data.summary.average_stats.{play,win_rate,pick_rate}",
    "data.starter_items.{ids[],ids_names[],pick_rate,play,win}",
    "data.boots.{ids[],ids_names[],pick_rate,play,win}",
    "data.core_items.{ids[],ids_names[],pick_rate,play,win}",
    "data.fourth_items[].{ids[],ids_names[],pick_rate,play,win}",
    "data.fifth_items[].{ids[],ids_names[],pick_rate,play,win}",
    "data.summoner_spells.{ids[],ids_names[],pick_rate,play,win}",
    "data.runes.{primary_page_name,primary_rune_names[],secondary_page_name,"
    "secondary_rune_names[],stat_mod_names[],pick_rate,play,win}",
    "data.skill_masteries.{ids[],pick_rate,play,win}",
    "data.trends",  # sa « version » donne le patch
]

# Rôles de Riot (match-v5) -> positions d'OP.GG.
POSITIONS = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "mid", "BOTTOM": "adc",
             "UTILITY": "support"}
# OP.GG renvoie parfois les sorts par numéro seulement.
SPELLS = {1: "Purge", 3: "Fatigue", 4: "Flash", 6: "Fantôme", 7: "Soins", 11: "Châtiment",
          12: "Téléportation", 13: "Clarté", 14: "Embrasement", 21: "Barrière"}
# Et les fragments de runes toujours par numéro (vérifié le 22/11/2026).
STAT_MODS = {5001: "PV croissants", 5005: "Vitesse d'attaque", 5007: "Accélération de compétence",
             5008: "Force adaptative", 5010: "Vitesse de déplacement", 5011: "PV",
             5013: "Ténacité et résistance aux ralentissements"}

Fetch = Callable[[dict[str, Any]], str]


class OpggError(Exception):
    """Réponse absente, trop grosse ou illisible."""


# --- Lecture du format texte d'OP.GG ---

_CLASS_RE = re.compile(r"^class\s+(\w+)\s*:\s*(.*)$")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_NAME_RE = re.compile(r"[A-Za-z_]\w*")


def parse_opgg_text(text: str) -> dict[str, Any] | None:
    """« class Stats: play,win » + « Resultat(Stats(10, 6)) » -> dictionnaires nommés.

    Une classe inconnue donne ses valeurs brutes. Renvoie None si le texte est illisible."""
    if not isinstance(text, str):
        return None
    classes: dict[str, list[str]] = {}
    body: list[str] = []
    for line in text.splitlines():
        found = _CLASS_RE.match(line.strip())
        if found:
            classes[found[1]] = [c.strip() for c in found[2].split(",") if c.strip()]
        elif line.strip():
            body.append(line)
    source = "\n".join(body)
    pos = 0

    def skip() -> None:
        nonlocal pos
        while pos < len(source) and source[pos].isspace():
            pos += 1

    def fail() -> None:
        raise OpggError(f"format OP.GG inattendu (position {pos})")

    def items(close: str, depth: int) -> list[Any]:
        nonlocal pos
        values: list[Any] = []
        skip()
        if pos < len(source) and source[pos] == close:
            pos += 1
            return values
        while True:
            values.append(value(depth + 1))
            skip()
            if pos < len(source) and source[pos] == ",":
                pos += 1
            elif pos < len(source) and source[pos] == close:
                pos += 1
                return values
            else:
                fail()

    def value(depth: int) -> Any:
        nonlocal pos
        if depth > MAX_DEPTH:
            fail()
        skip()
        if pos >= len(source):
            fail()
        char = source[pos]
        if char == "[":
            pos += 1
            return items("]", depth)
        if char in "\"'":
            pos += 1
            out = []
            while pos < len(source) and source[pos] != char:
                if source[pos] == "\\" and pos + 1 < len(source):
                    pos += 1
                out.append(source[pos])
                pos += 1
            if pos >= len(source):
                fail()
            pos += 1
            return "".join(out)
        number = _NUMBER_RE.match(source, pos)
        if number:
            pos = number.end()
            return float(number[0]) if any(c in number[0] for c in ".eE") else int(number[0])
        name = _NAME_RE.match(source, pos)
        if not name:
            fail()
        assert name is not None
        pos = name.end()
        if name[0] in ("None", "null"):
            return None
        if name[0] in ("True", "true"):
            return True
        if name[0] in ("False", "false"):
            return False
        skip()
        if pos >= len(source) or source[pos] != "(":
            fail()
        pos += 1
        args = items(")", depth)
        fields = classes.get(name[0])
        if fields is None:
            return args[0] if len(args) == 1 else args
        return dict(zip(fields, args, strict=False))

    try:
        result = value(0)
    except (OpggError, RecursionError):
        return None
    return result if isinstance(result, dict) else None


# --- Mise en forme ---

def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _rate(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1:
        return round(float(value), 3)
    return None


def _names(raw: dict[str, Any], names_key: str = "ids_names") -> list[str]:
    names = raw.get(names_key)
    return [str(n) for n in names if isinstance(n, (str, int))] if isinstance(names, list) else []


def _option(raw: Any, spells: bool = False) -> dict[str, Any] | None:
    """Une option de build : objets (ou sorts), parties, taux de victoire et de choix."""
    if not isinstance(raw, dict):
        return None
    names = _names(raw)
    if spells:  # OP.GG donne souvent les sorts par numéro, sans nom
        ids = raw.get("ids")
        names = [SPELLS.get(i, str(i)) for i in (ids if isinstance(ids, list) else [])
                 if isinstance(i, int)]
    if not names:
        return None
    play = _count(raw.get("play"))
    return {"objets" if not spells else "sorts": names, "parties": play,
            "taux_victoire": round(_count(raw.get("win")) / play, 3) if play else None,
            "taux_choix": _rate(raw.get("pick_rate"))}


def _options(raw: Any) -> list[dict[str, Any]]:
    values = raw if isinstance(raw, list) else [raw]
    return [o for o in (_option(v) for v in values) if o is not None]


def _runes(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    primary = _names(raw, "primary_rune_names")
    if not primary:
        return None
    play = _count(raw.get("play"))
    return {"principale": str(raw.get("primary_page_name") or ""), "runes": primary,
            "secondaire": str(raw.get("secondary_page_name") or ""),
            "runes_secondaires": _names(raw, "secondary_rune_names"),
            "fragments": [STAT_MODS.get(int(f), f) if str(f).isdigit() else f
                          for f in _names(raw, "stat_mod_names")], "parties": play,
            "taux_victoire": round(_count(raw.get("win")) / play, 3) if play else None,
            "taux_choix": _rate(raw.get("pick_rate"))}


def _skills(raw: Any) -> list[str] | None:
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if not isinstance(raw, dict) or not isinstance(raw.get("ids"), list):
        return None
    keys = [str(k) for k in raw["ids"] if isinstance(k, str) and k in "QWER" and k]
    return keys or None


def _patch(raw: Any) -> str | None:
    """Patch des statistiques. OP.GG donne la version interne du jeu (« 16.23 ») ; depuis
    2025, le patch affiché aux joueurs a 10 de plus (« 26.23 »)."""
    if not isinstance(raw, dict):
        return None
    wins = raw.get("win")
    versions = [w.get("version") for w in (wins if isinstance(wins, list) else [wins])
                if isinstance(w, dict)]
    found = [v for v in versions if isinstance(v, str) and re.fullmatch(r"\d{1,2}\.\d{1,2}", v)]
    if not found:
        return None
    major, minor = max(tuple(int(x) for x in v.split(".")) for v in found)
    return f"{major + 10 if 15 <= major < 25 else major}.{minor}"


def build_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    """Réponse lue d'OP.GG -> build propre. Lève OpggError s'il n'y a pas d'objets."""
    data = parsed.get("data")
    if not isinstance(data, dict):
        raise OpggError("réponse d'OP.GG sans données")
    summary = data.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    stats = summary.get("average_stats")
    stats = stats if isinstance(stats, dict) else {}
    core = _option(data.get("core_items"))
    if core is None:
        raise OpggError("pas assez de parties sur OP.GG pour ce champion et ce rôle")
    return {
        "source": "op.gg (classée, toutes divisions)",
        "patch": _patch(data.get("trends")),
        "parties": _count(stats.get("play")),
        "taux_victoire": _rate(stats.get("win_rate")),
        "taux_choix": _rate(stats.get("pick_rate")),
        "depart": _option(data.get("starter_items")),
        "bottes": _option(data.get("boots")),
        "objets_principaux": core,
        "quatrieme_objet": _options(data.get("fourth_items"))[:3],
        "cinquieme_objet": _options(data.get("fifth_items"))[:3],
        "sorts": _option(data.get("summoner_spells"), spells=True),
        "runes": _runes(data.get("runes")),
        "competences_max": _skills(data.get("skill_masteries")),
    }


# --- Appel à OP.GG ---

def opgg_champion(champion: str) -> str:
    """« Bel'Veth », « Belveth » ou « MonkeyKing » -> « BELVETH », « MONKEYKING »."""
    return re.sub(r"[^A-Za-z0-9]", "", champion or "").upper()


def _http_fetch(payload: dict[str, Any]) -> str:
    """POST vers OP.GG, réponse limitée à MAX_BYTES."""
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT,
               "Accept": "application/json, text/event-stream"}
    with httpx.Client(timeout=TIMEOUT_S, follow_redirects=False) as client, \
            client.stream("POST", OPGG_URL, json=payload, headers=headers) as response:
        if response.status_code != 200:
            raise OpggError(f"OP.GG a répondu {response.status_code}")
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_BYTES:
                raise OpggError("réponse d'OP.GG trop grosse")
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _content_text(raw: str) -> str:
    """Texte utile d'une réponse JSON-RPC (JSON brut ou flux « data: {...} »)."""
    line = next((li for li in raw.splitlines() if li.startswith("data:")), None)
    try:
        message = json.loads(line[5:] if line else raw)
    except ValueError as exc:
        raise OpggError("réponse illisible d'OP.GG") from exc
    if not isinstance(message, dict):
        raise OpggError("réponse illisible d'OP.GG")
    if message.get("error"):
        error = message["error"]
        detail = error.get("message") if isinstance(error, dict) else str(error)
        raise OpggError(f"OP.GG ne connaît pas ce champion ou ce rôle ({str(detail)[:120]})")
    result = message.get("result")
    content = result.get("content") if isinstance(result, dict) else None
    if not isinstance(content, list):
        raise OpggError("réponse d'OP.GG sans contenu")
    return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))


def _read_cache(path: Path, now: float) -> dict[str, Any] | None:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (isinstance(saved, dict) and isinstance(saved.get("enregistre"), (int, float))
            and 0 <= now - saved["enregistre"] < CACHE_TTL_S
            and isinstance(saved.get("build"), dict)):
        return dict(saved["build"])
    return None


def _write_cache(path: Path, build: dict[str, Any], now: float) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"enregistre": now, "build": build}, ensure_ascii=False),
                        encoding="utf-8")
    except OSError:
        pass  # sans cache, on rappellera OP.GG la prochaine fois


def enabled_from_env() -> bool:
    return os.environ.get("LOLCOACH_OPGG", "1").strip().lower() not in ("0", "non", "aucun", "no")


def champion_build(champion: str, role: str, *, fetch: Fetch | None = None,
                   cache_dir: Path | None = None, now: Callable[[], float] = time.time,
                   enabled: bool | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Build OP.GG du champion à ce rôle : (build, None), ou (None, raison). Jamais d'exception."""
    try:
        if not (enabled_from_env() if enabled is None else enabled):
            return None, "OP.GG désactivé (LOLCOACH_OPGG=0)"
        name = opgg_champion(champion)
        position = POSITIONS.get((role or "").upper())
        if not re.fullmatch(r"[A-Z0-9]{2,30}", name) or position is None:
            return None, f"champion ou rôle inconnu ({champion!r}, {role!r})"
        path = (cache_dir or CACHE_DIR) / f"{name}_{position}.json"
        moment = now()
        cached = _read_cache(path, moment)
        if cached is not None:
            return cached, None
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": TOOL, "arguments": {
                       "champion": name, "position": position, "game_mode": "ranked",
                       "lang": LANG, "desired_output_fields": FIELDS}}}
        parsed = parse_opgg_text(_content_text((fetch or _http_fetch)(payload)))
        if parsed is None:
            raise OpggError("format de réponse d'OP.GG inattendu")
        build = build_from_parsed(parsed)
        build["recupere_le"] = datetime.fromtimestamp(moment, UTC).strftime("%d/%m/%Y")
        _write_cache(path, build, moment)
        return build, None
    except OpggError as exc:
        return None, str(exc)
    except httpx.HTTPError as exc:
        return None, f"OP.GG injoignable ({type(exc).__name__})"
    except Exception as exc:  # noqa: BLE001 - le coach doit continuer sans build
        return None, f"erreur inattendue en lisant OP.GG ({type(exc).__name__})"


def _percent(value: float | None) -> str:
    return "?" if value is None else f"{value * 100:.0f} %".replace(".", ",")


def render_build(champion: str, role: str, build: dict[str, Any]) -> list[str]:
    """Le build en quelques lignes lisibles (commande « lolcoach-coach build »)."""
    patch = f", patch {build['patch']}" if build.get("patch") else ""
    lines = [f"Build {champion} ({role.lower()}) d'après op.gg{patch} : "
             f"{build.get('parties', 0)} parties, {_percent(build.get('taux_victoire'))} de "
             "victoires."]

    def option(label: str, opt: dict[str, Any] | None, key: str = "objets") -> None:
        if opt:
            lines.append(f"- {label} : {', '.join(opt[key])} ({_percent(opt.get('taux_victoire'))}"
                         f" de victoires, choisi {_percent(opt.get('taux_choix'))})")

    option("Départ", build.get("depart"))
    option("Objets principaux", build.get("objets_principaux"))
    option("Bottes", build.get("bottes"))
    for label, key in (("4e objet", "quatrieme_objet"), ("5e objet", "cinquieme_objet")):
        choices = build.get(key) or []
        if choices:
            lines.append(f"- {label} : " + " ; ".join(
                f"{', '.join(c['objets'])} ({_percent(c.get('taux_victoire'))})" for c in choices))
    option("Sorts", build.get("sorts"), "sorts")
    runes = build.get("runes")
    if runes:
        shards = f", fragments {', '.join(runes['fragments'])}" if runes["fragments"] else ""
        lines.append(f"- Runes : {runes['principale']} ({', '.join(runes['runes'])}), "
                     f"{runes['secondaire']} ({', '.join(runes['runes_secondaires'])})"
                     f"{shards} ({_percent(runes.get('taux_victoire'))} de victoires)")
    if build.get("competences_max"):
        lines.append(f"- Compétences à monter : {' > '.join(build['competences_max'])}")
    return lines
