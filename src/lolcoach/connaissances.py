"""Fiches de connaissances : ce que le coach « sait », sous forme de fichiers lisibles.

Chaque fiche est un fichier Markdown, rangé dans le dossier des connaissances (par défaut
`connaissances/`, réglable par LOLCOACH_KNOWLEDGE_DIR), avec un en-tête :

    ---
    id: jgl-setup-drake
    titre: Préparer un dragon
    role: JUNGLE
    theme: objectifs
    champions: []
    patch_min: "16.20"
    source: youtube:abc123
    statut: a_valider
    ---
    Être sur place 45 s avant l'apparition…

Pourquoi des fichiers plutôt qu'une base vectorielle :
- le joueur peut relire et corriger ce que l'IA a « appris » : une fiche n'est utilisée
  par le coach qu'une fois passée en `valide` ;
- Git garde l'historique ; une fiche dépassée par un patch passe en `obsolete` ;
- quelques centaines de fiches tiennent dans la mémoire d'un modèle de langage : un filtre
  par rôle, thème et champion, plus une recherche par mots, suffit.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DIR = "connaissances"
TO_VALIDATE, VALID, OBSOLETE = "a_valider", "valide", "obsolete"
STATUSES = (TO_VALIDATE, VALID, OBSOLETE)
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY", "TOUS")
MAX_CARD_BYTES = 50_000
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,79}$")
_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


class KnowledgeError(ValueError):
    """Fiche illisible ou invalide."""


def knowledge_dir() -> Path:
    return Path(os.environ.get("LOLCOACH_KNOWLEDGE_DIR", "").strip() or DEFAULT_DIR)


@dataclass
class Card:
    id: str
    title: str
    role: str
    theme: str
    body: str
    status: str = TO_VALIDATE
    champions: list[str] = field(default_factory=list)
    patch_min: str | None = None
    source: str | None = None
    path: Path | None = None

    def as_dict(self, with_body: bool = True) -> dict[str, object]:
        data: dict[str, object] = {
            "id": self.id, "titre": self.title, "role": self.role, "theme": self.theme,
            "champions": self.champions, "patch_min": self.patch_min, "source": self.source,
            "statut": self.status,
        }
        if with_body:
            data["contenu"] = self.body
        return data

    def to_markdown(self) -> str:
        champions = ", ".join(self.champions)
        lines = ["---", f"id: {self.id}", f"titre: {_one_line(self.title)}",
                 f"role: {self.role}", f"theme: {_one_line(self.theme)}",
                 f"champions: [{champions}]"]
        if self.patch_min:
            lines.append(f'patch_min: "{_one_line(self.patch_min)}"')
        if self.source:
            lines.append(f"source: {_one_line(self.source)}")
        lines += [f"statut: {self.status}", "---", self.body.strip(), ""]
        return "\n".join(lines)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())[:200]


def _value(raw: str) -> str | list[str]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return [item.strip().strip("\"'") for item in raw[1:-1].split(",") if item.strip()]
    return raw.strip("\"'")


def parse_card(text: str, path: Path | None = None) -> Card:
    found = _FRONT_RE.match(text.lstrip("﻿"))
    if not found:
        raise KnowledgeError(f"{path or 'fiche'} : en-tête --- manquant")
    header: dict[str, str | list[str]] = {}
    for line in found.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, raw = line.partition(":")
        if not sep:
            raise KnowledgeError(f"{path or 'fiche'} : ligne d'en-tête illisible {line!r}")
        header[key.strip().lower()] = _value(raw)

    def text_field(name: str, default: str | None = None) -> str | None:
        value = header.get(name, default)
        if isinstance(value, list):
            raise KnowledgeError(f"{path or 'fiche'} : {name} doit être un texte")
        return value or default

    card_id = text_field("id") or (path.stem if path else "")
    if not _ID_RE.match(card_id):
        raise KnowledgeError(f"{path or 'fiche'} : identifiant invalide {card_id!r} "
                             "(minuscules, chiffres et tirets)")
    role = (text_field("role", "TOUS") or "TOUS").upper()
    status = (text_field("statut", TO_VALIDATE) or TO_VALIDATE).lower()
    if role not in ROLES:
        raise KnowledgeError(f"{path or 'fiche'} : rôle inconnu {role!r}")
    if status not in STATUSES:
        raise KnowledgeError(f"{path or 'fiche'} : statut inconnu {status!r}")
    champions = header.get("champions") or []
    if isinstance(champions, str):
        champions = [champions]
    body = found.group(2).strip()
    if not body:
        raise KnowledgeError(f"{path or 'fiche'} : fiche vide")
    return Card(
        id=card_id, title=text_field("titre") or card_id, role=role,
        theme=(text_field("theme", "general") or "general").lower(), body=body, status=status,
        champions=[c for c in champions if c], patch_min=text_field("patch_min"),
        source=text_field("source"), path=path,
    )


def load_cards(folder: Path | None = None) -> tuple[list[Card], list[str]]:
    """Toutes les fiches du dossier (sous-dossiers compris), et les erreurs de lecture."""
    folder = folder or knowledge_dir()
    cards, problems = [], []
    if not folder.is_dir():
        return [], []
    seen: set[str] = set()
    for path in sorted(folder.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            if path.stat().st_size > MAX_CARD_BYTES:
                raise KnowledgeError(f"{path} : fiche trop longue")
            card = parse_card(path.read_text(encoding="utf-8"), path)
        except (OSError, UnicodeDecodeError, KnowledgeError) as exc:
            problems.append(str(exc))
            continue
        if card.id in seen:
            problems.append(f"{path} : identifiant {card.id!r} déjà utilisé")
            continue
        seen.add(card.id)
        cards.append(card)
    return cards, problems


def _words(text: str) -> set[str]:
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return {w for w in re.findall(r"[a-z0-9]+", plain) if len(w) > 2}


def search(cards: list[Card], role: str | None = None, theme: str | None = None,
           champion: str | None = None, query: str | None = None,
           statuses: tuple[str, ...] = (VALID,), limit: int = 10) -> list[Card]:
    """Fiches correspondant aux filtres, les plus pertinentes d'abord.

    Un rôle garde aussi les fiches « TOUS » ; un champion garde aussi les fiches générales
    (sans champion). Avec `query`, les fiches sont classées par mots en commun."""
    wanted = _words(query or "")
    scored = []
    for card in cards:
        if card.status not in statuses:
            continue
        if role and card.role not in (role.upper(), "TOUS"):
            continue
        if theme and card.theme != theme.lower():
            continue
        if champion and card.champions and champion.lower() not in (
                c.lower() for c in card.champions):
            continue
        score = 0.0
        if wanted:
            title, body = _words(card.title), _words(card.body)
            score = 3 * len(wanted & title) + len(wanted & body)
            if score == 0:
                continue
        if champion and card.champions:
            score += 2  # une fiche propre au champion passe avant une fiche générale
        scored.append((-score, card.id, card))
    return [card for _, _, card in sorted(scored)[:limit]]


def slugify(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9]+", "-", plain).strip("-")[:60]
    return slug or "fiche"


def write_card(card: Card, folder: Path | None = None, overwrite: bool = False) -> Path:
    """Écrit une fiche (proposée par une IA ou par le joueur). Rangée par rôle."""
    if not _ID_RE.match(card.id):
        raise KnowledgeError(f"identifiant invalide {card.id!r}")
    folder = folder or knowledge_dir()
    target = folder / card.role.lower() / f"{card.id}.md"
    if target.exists() and not overwrite:
        raise KnowledgeError(f"la fiche {card.id} existe déjà ({target})")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = card.to_markdown()
    parse_card(text)  # ce qu'on écrit doit pouvoir être relu
    target.write_text(text, encoding="utf-8")
    return target


def set_status(card_id: str, status: str, folder: Path | None = None) -> Path:
    if status not in STATUSES:
        raise KnowledgeError(f"statut inconnu {status!r}")
    cards, _ = load_cards(folder)
    card = next((c for c in cards if c.id == card_id), None)
    if card is None or card.path is None:
        raise KnowledgeError(f"fiche {card_id!r} introuvable")
    card.status = status
    card.path.write_text(card.to_markdown(), encoding="utf-8")
    return card.path
