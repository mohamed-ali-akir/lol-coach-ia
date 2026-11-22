"""Les consignes données aux modèles d'IA, et les schémas JSON de leurs réponses.

Regroupées ici pour être relues et améliorées au même endroit. `PROMPT_VERSION` est
enregistré avec chaque rapport : on saura quelle consigne a produit quel texte.
"""

from __future__ import annotations

import json
from typing import Any

from lolcoach.client_lol.replay import Moment
from lolcoach.suivi.catalogue import AI, CATALOGUE

PROMPT_VERSION = "2026-11-1"

_ERROR_ITEM = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "un code du catalogue fourni"},
        "minute": {"type": "string", "description": "minute de jeu, format 14:22"},
        "confiance": {"type": "number", "minimum": 0, "maximum": 1},
        "preuve": {"type": "string", "description": "ce qui se voit sur les images"},
    },
    "required": ["code", "confiance", "preuve"],
}

# Même forme que ce qu'accepte `suivi.store.record_ai_analysis`.
VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "minute": {"type": "string"},
                    "resume": {"type": "string", "description": "ce qui se passe, en une phrase"},
                    "erreurs": {"type": "array", "items": _ERROR_ITEM},
                },
                "required": ["minute", "resume", "erreurs"],
            },
        },
        "propositions": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"nom": {"type": "string"},
                                     "description": {"type": "string"}},
                      "required": ["nom", "description"]},
        },
    },
    "required": ["moments"],
}

CARDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fiches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "titre": {"type": "string"},
                    "role": {"type": "string",
                             "enum": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY", "TOUS"]},
                    "theme": {"type": "string"},
                    "champions": {"type": "array", "items": {"type": "string"}},
                    "contenu": {"type": "string"},
                },
                "required": ["titre", "role", "theme", "contenu"],
            },
        },
    },
    "required": ["fiches"],
}

CARD_THEMES = ("pathing", "objectifs", "ganks", "tracking", "vision", "invade", "farm",
               "combat", "macro", "mental", "general")


def _ai_catalogue(role: str) -> str:
    return "\n".join(f"- {e.code} : {e.label}. {e.description}" for e in CATALOGUE
                     if role in e.roles and e.detected_by == AI)


def vision_prompt(moment: Moment, champion: str, role: str, facts: dict[str, Any]) -> str:
    """Consigne pour regarder les images d'un moment de la partie."""
    return f"""Tu es un coach League of Legends qui analyse le replay d'un joueur.
Le joueur joue {champion} ({role}). Les images sont des captures du replay prises une fois
par seconde, de 20 secondes avant à 8 secondes après {_clock(moment.ts_ms)}, caméra sur son
champion, avec le brouillard de guerre de son équipe : tu vois ce qu'il pouvait voir.
Pourquoi ce moment a été retenu : {moment.reason}.

Faits établis par les données de la partie (fiables, ne les contredis pas) :
{json.dumps(facts, ensure_ascii=False, indent=1)}

Regarde la mini-carte, la vie et les sorts des champions, les vagues de sbires et les balises.
Décris le moment en une phrase, puis relève les erreurs du joueur que les images MONTRENT,
uniquement avec ces codes :
{_ai_catalogue(role)}

Règles :
- une erreur seulement si elle se voit ; en cas de doute, confiance basse ou rien ;
- une erreur importante absente de la liste va dans « propositions », jamais dans « erreurs » ;
- « preuve » dit ce qu'on voit (ex. « vague adverse sous leur tour, Flash adverse disponible »).
Réponds uniquement par un objet JSON de cette forme :
{json.dumps(VISION_SCHEMA, ensure_ascii=False)}"""


def report_prompt(context: dict[str, Any]) -> str:
    """Consigne pour rédiger le rapport de coaching complet."""
    return f"""Tu es le coach League of Legends de ce joueur. Rédige son rapport d'après-partie en
français, comme un coach humain qui a regardé la partie et qui le suit depuis plusieurs
semaines.

Règles :
- appuie-toi UNIQUEMENT sur les données ci-dessous ; n'invente ni minute, ni chiffre ;
- les positions ne sont connues qu'une fois par minute : dis « probablement » pour une
  déduction ;
- 2 ou 3 axes de travail au maximum, du plus important au moins important. Pour chacun :
  la minute, ce qui s'est passé, ce qu'il fallait faire à la place, et un exercice précis ;
- utilise la progression : signale une erreur qui revient (« troisième partie sur cinq… »)
  et félicite une erreur corrigée ou en progrès ;
- tiens compte des notes du joueur (son contexte) et appuie-toi sur les fiches validées ;
- termine par un point positif sincère et l'objectif de la prochaine partie ;
- pas plus de 450 mots, titres courts, pas de tableau.

Données :
{json.dumps(context, ensure_ascii=False, indent=1)}"""


def cards_prompt(role: str, url: str | None = None) -> str:
    """Consigne pour transformer une vidéo éducative en fiches de connaissances."""
    source = f"la vidéo {url}" if url else "la vidéo jointe"
    return f"""Regarde {source} : c'est une vidéo éducative League of Legends.
Transforme ce qu'elle enseigne en fiches de connaissances pour un coach, rôle principal :
{role}.

Règles :
- une fiche = un concept actionnable (quand, pourquoi, comment), 80 à 200 mots, en français ;
- décris ce que le coach MONTRE à l'écran (mini-carte, vagues, timers), pas seulement ce
  qu'il dit ;
- ignore l'introduction, la publicité et les anecdotes ;
- précise les champions seulement si le conseil leur est propre (sinon liste vide) ;
- thème parmi : {", ".join(CARD_THEMES)} ;
- au plus 8 fiches, sans doublon.
Réponds uniquement par un objet JSON de cette forme :
{json.dumps(CARDS_SCHEMA, ensure_ascii=False)}"""


def _clock(ts_ms: int) -> str:
    seconds = max(0, ts_ms) // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"
