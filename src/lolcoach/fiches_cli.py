"""Les fiches de connaissances du coach : créer à partir de vidéos, relire, valider.

Exemples :
    lolcoach-fiches lister                       # toutes les fiches et leur statut
    lolcoach-fiches lister --statut a_valider
    lolcoach-fiches video "https://www.youtube.com/watch?v=…" --role JUNGLE
    lolcoach-fiches consigne --role JUNGLE       # consigne à coller dans l'appli Gemini
    lolcoach-fiches importer reponse.json --source youtube:abc123
    lolcoach-fiches valider jgl-setup-drake      # le coach pourra s'en servir
    lolcoach-fiches obsolete jgl-setup-drake     # dépassée par un patch

Une fiche créée par une IA est « à valider » : le coach ne l'utilise qu'après ta relecture.
`video` passe par l'API Gemini (LOLCOACH_FICHES=gemini, clé GEMINI_API_KEY) ou par une
commande (LOLCOACH_FICHES=agy, ou commande avec LOLCOACH_FICHES_CMD).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from lolcoach.connaissances import (
    ROLES,
    STATUSES,
    TO_VALIDATE,
    Card,
    KnowledgeError,
    load_cards,
    set_status,
    slugify,
    write_card,
)
from lolcoach.ia.consignes import CARD_THEMES, CARDS_SCHEMA, cards_prompt
from lolcoach.ia.fournisseurs import (
    DEFAULT_GEMINI_MODEL,
    PRESETS,
    AIError,
    CommandProvider,
    GeminiProvider,
    extract_json,
)

_YOUTUBE_RE = re.compile(
    r"^https://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})(?:[&?#][^\s]*)?$")
MAX_CARDS = 8


def youtube_id(url: str) -> str:
    found = _YOUTUBE_RE.match(url.strip())
    if not found:
        raise KnowledgeError(f"lien YouTube non reconnu : {url!r}")
    return found.group(1)


def cards_from_answer(answer: dict[str, Any], source: str, default_role: str,
                      existing: set[str]) -> list[Card]:
    """Fiches tirées d'une réponse JSON d'IA ({"fiches": [...]}), validées une à une."""
    raw_cards = answer.get("fiches")
    if not isinstance(raw_cards, list):
        raise KnowledgeError("réponse sans liste « fiches »")
    cards = []
    for raw in raw_cards[:MAX_CARDS]:
        if not isinstance(raw, dict):
            continue
        title = " ".join(str(raw.get("titre") or "").split())[:120]
        body = str(raw.get("contenu") or "").strip()
        if not title or len(body) < 40:
            continue
        role = str(raw.get("role") or default_role).upper()
        role = role if role in ROLES else default_role
        theme = str(raw.get("theme") or "general").lower()
        theme = theme if theme in CARD_THEMES else "general"
        champions = [" ".join(str(c).split())[:30] for c in raw.get("champions") or []
                     if str(c).strip()][:5]
        prefix = {"JUNGLE": "jgl", "TOP": "top", "MIDDLE": "mid", "BOTTOM": "adc",
                  "UTILITY": "sup"}.get(role, "lol")
        base = f"{prefix}-{slugify(title)}"[:70].rstrip("-")
        card_id, n = base, 2
        while card_id in existing:
            card_id, n = f"{base}-{n}", n + 1
        existing.add(card_id)
        cards.append(Card(id=card_id, title=title, role=role, theme=theme, body=body[:4000],
                          status=TO_VALIDATE, champions=champions, source=source[:200]))
    if not cards:
        raise KnowledgeError("aucune fiche exploitable dans la réponse")
    return cards


def _cards_provider() -> GeminiProvider | CommandProvider:
    choice = os.environ.get("LOLCOACH_FICHES", "gemini").strip().lower() or "gemini"
    if choice == "gemini":
        return GeminiProvider(os.environ.get("LOLCOACH_FICHES_MODEL", "").strip()
                              or DEFAULT_GEMINI_MODEL)
    template = os.environ.get("LOLCOACH_FICHES_CMD", "").strip()
    if choice == "agy":
        template = template or PRESETS["agy"]["vision"]
    if not template:
        raise AIError("LOLCOACH_FICHES_CMD est vide : décris la commande à lancer")
    return CommandProvider(template, name=choice)


def _save(cards: list[Card]) -> None:
    for card in cards:
        path = write_card(card)
        print(f"+ {card.id} ({card.theme}) : {card.title}  →  {path}")
    print(f"\n{len(cards)} fiche(s) à relire, puis : lolcoach-fiches valider <id>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lolcoach-fiches", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="commande", required=True)
    p_list = sub.add_parser("lister", help="fiches et statuts")
    p_list.add_argument("--statut", choices=STATUSES)
    p_list.add_argument("--role", type=str.upper, choices=ROLES)
    p_video = sub.add_parser("video", help="créer des fiches à partir d'une vidéo YouTube")
    p_video.add_argument("url")
    p_video.add_argument("--role", type=str.upper, choices=ROLES, default="JUNGLE")
    p_prompt = sub.add_parser("consigne", help="consigne à coller dans l'appli Gemini")
    p_prompt.add_argument("url", nargs="?")
    p_prompt.add_argument("--role", type=str.upper, choices=ROLES, default="JUNGLE")
    p_import = sub.add_parser("importer", help="importer la réponse JSON de l'IA")
    p_import.add_argument("fichier", type=Path)
    p_import.add_argument("--source", default="manuel")
    p_import.add_argument("--role", type=str.upper, choices=ROLES, default="JUNGLE")
    for name, help_text in (("valider", "le coach pourra s'en servir"),
                            ("obsolete", "dépassée (patch, méta)"),
                            ("a-revoir", "remettre « à valider »")):
        sub.add_parser(name, help=help_text).add_argument("id")
    return parser


def _run(args: argparse.Namespace) -> None:
    load_dotenv()
    if args.commande == "lister":
        cards, problems = load_cards()
        for card in cards:
            if (args.statut and card.status != args.statut) or (
                    args.role and card.role != args.role):
                continue
            print(f"{card.status:<10} {card.role:<8} {card.theme:<10} {card.id}  {card.title}")
        for problem in problems:
            print(f"! {problem}", file=sys.stderr)
        if not cards:
            print("Aucune fiche. Essaie : lolcoach-fiches video <lien YouTube>")
        return
    if args.commande in ("valider", "obsolete", "a-revoir"):
        status = {"valider": "valide", "obsolete": "obsolete", "a-revoir": TO_VALIDATE}
        print(f"{set_status(args.id, status[args.commande])} : {status[args.commande]}")
        return
    if args.commande == "consigne":
        print(cards_prompt(args.role, args.url))
        print("\n---\nEnregistre la réponse dans un fichier .json, puis : "
              "lolcoach-fiches importer reponse.json --source youtube:<id>")
        return
    existing = {card.id for card in load_cards()[0]}
    if args.commande == "importer":
        answer = extract_json(args.fichier.read_text(encoding="utf-8"))
        _save(cards_from_answer(answer, args.source, args.role, existing))
        return
    video = youtube_id(args.url)
    provider = _cards_provider()
    print(f"Analyse de la vidéo par {provider.name}…")
    if isinstance(provider, GeminiProvider):
        answer = provider.look_video(cards_prompt(args.role), args.url, CARDS_SCHEMA)
    else:
        answer = extract_json(provider.write(cards_prompt(args.role, args.url)))
    _save(cards_from_answer(answer, f"youtube:{video}", args.role, existing))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        _run(args)
    except (KnowledgeError, AIError, OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Erreur : {exc}")


if __name__ == "__main__":
    main()
