"""Le catalogue des erreurs suivies : des codes fixes, pour pouvoir compter.

Pourquoi un catalogue plutôt que du texte libre : si l'IA écrit « tu t'es fait surprendre
sans vision » une fois et « facecheck dans la rivière » la suivante, impossible de savoir
que c'est la même erreur. Chaque erreur a donc un code, et une IA qui analyse une partie
doit choisir dans cette liste. Une erreur qu'elle croit nouvelle devient une
*proposition*, que le joueur examine ; elle n'entre jamais seule dans le catalogue.

Chaque erreur appartient à un type de situation. Le taux d'erreur est le nombre d'erreurs
rapporté au nombre de fois où la situation s'est présentée : « 1 dragon raté sur 4 » ne
dit pas la même chose que « 1 dragon raté sur 1 ».

Ajouter une erreur = ajouter une entrée à `CATALOGUE`. Si elle se détecte dans la timeline,
écrire sa règle dans `detection.py` et augmenter `RULES_VERSION`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Types de situations
GAME = "PARTIE"  # une par partie
EPIC_OBJECTIVE = "OBJECTIF"  # une par monstre épique pris (par l'une ou l'autre équipe)
KEY_MOMENT = "MOMENT_CLE"  # une par moment analysé en vidéo par une IA

SITUATION_LABELS = {
    GAME: "parties",
    EPIC_OBJECTIVE: "monstres épiques",
    KEY_MOMENT: "moments analysés en vidéo",
}

RULE, AI = "regle", "ia"  # comment l'erreur est repérée
ALL_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


@dataclass(frozen=True)
class ErrorType:
    code: str
    label: str
    situation: str
    detected_by: str  # RULE (timeline) ou AI (captures, clips, analyse d'un modèle)
    description: str  # ce qui caractérise l'erreur, pour l'IA comme pour le joueur
    advice: str  # quoi faire à la place
    roles: tuple[str, ...] = ALL_ROLES
    theme: str = "jungle"


CATALOGUE: tuple[ErrorType, ...] = (
    # --- Repérées dans la timeline ---
    ErrorType(
        "MORT_ISOLEE", "Mort seul face à plusieurs adversaires", GAME, RULE,
        "Mort face à au moins deux adversaires, sans aucun allié à proximité.",
        "Avant d'avancer dans une zone sans vision, regarde la mini-carte : si les adversaires "
        "manquants peuvent être là, recule ou attends un allié.", theme="survie"),
    ErrorType(
        "MORT_GANK_SUBI", "Gank subi", GAME, RULE,
        "Mort avant 15 minutes face à au moins deux adversaires dont leur jungler.",
        "Pose une balise en rivière avant de pousser, et recule dès que le jungler adverse "
        "n'est plus visible.", theme="survie"),
    ErrorType(
        "MORT_AVANT_OBJECTIF", "Mort juste avant un objectif adverse", EPIC_OBJECTIVE, RULE,
        "Mort dans les 90 secondes qui précèdent un monstre épique pris par l'adversaire.",
        "Une minute avant un objectif, ne prends plus de risque : fais ton retour à la base "
        "et arrive plein de vie avec ton équipe."),
    ErrorType(
        "OBJ_ABSENT", "Absent d'un objectif pris par l'adversaire", EPIC_OBJECTIVE, RULE,
        "Vivant mais loin (plus de 3 500 unités) 30 secondes avant un monstre épique pris "
        "par l'adversaire.",
        "Sois sur place 30 s avant l'apparition, ou prends un objectif de l'autre côté de la "
        "carte au même moment.", roles=("JUNGLE",)),
    ErrorType(
        "JGL_CLEAR_LENT", "Premier clear lent", GAME, RULE,
        "Moins de monstres tués à 3:30 que le quart le plus lent des meilleurs joueurs sur "
        "ce champion.",
        "Enchaîne les camps sans détour ; chronomètre ton clear en outil d'entraînement.",
        roles=("JUNGLE",)),
    ErrorType(
        "JGL_FARM_FAIBLE", "Farm de jungle trop faible à 10 minutes", GAME, RULE,
        "Moins de monstres tués à 10 minutes que le quart le plus bas des meilleurs joueurs "
        "sur ce champion.",
        "Reprends tes camps dès leur réapparition entre deux actions.", roles=("JUNGLE",)),
    ErrorType(
        "JGL_SANS_IMPACT_EARLY", "Aucun impact avant 8 minutes", GAME, RULE,
        "Aucun kill ni assistance avant 8 minutes alors que ton équipe en a fait au moins deux.",
        "Après ton premier clear, cherche la lane dont la vague pousse vers ton allié et dont "
        "l'adversaire n'a plus son sort de fuite.", roles=("JUNGLE",)),
    ErrorType(
        "JGL_RETARD_FARM_ADVERSE", "Grand retard de farm sur le jungler adverse", GAME, RULE,
        "Au moins 15 monstres de retard sur le jungler adverse à 15 minutes.",
        "Vole ses camps quand il se montre de l'autre côté de la carte, et joue du côté de ta "
        "lane la plus forte.", roles=("JUNGLE",)),
    # --- Repérées par une IA sur les captures ou les clips du replay ---
    ErrorType(
        "MORT_SANS_VISION", "Mort dans une zone sans vision", KEY_MOMENT, AI,
        "Mort en entrant dans une zone sans balise alliée alors que des adversaires étaient "
        "invisibles sur la carte.",
        "Pose ou demande une balise avant d'entrer, ou passe par une zone visible.",
        theme="vision"),
    ErrorType(
        "JGL_GANK_SANS_PRIO", "Gank sans priorité", KEY_MOMENT, AI,
        "Gank lancé alors que la vague poussait vers la tour adverse ou que l'adversaire "
        "avait son sort de fuite.",
        "Gank quand la vague pousse vers ton allié et que l'adversaire est sans sort de fuite."),
    ErrorType(
        "JGL_FACECHECK", "Entrée dans un buisson sans information", KEY_MOMENT, AI,
        "Entrée dans un buisson ou une zone de jungle non visible, suivie d'un combat perdu.",
        "Utilise une balise, une compétence ou un allié pour vérifier avant d'entrer.",
        theme="vision"),
    ErrorType(
        "JGL_PAS_DE_TRACKING", "Jungler adverse non suivi", KEY_MOMENT, AI,
        "Décision prise sans tenir compte de la dernière position connue du jungler adverse "
        "(gank subi par un allié, invasion punie, objectif volé).",
        "Note la dernière apparition du jungler adverse et déduis son trajet probable avant "
        "chaque décision."),
    ErrorType(
        "JGL_RESET_MAL_PLACE", "Retour à la base mal placé", KEY_MOMENT, AI,
        "Retour à la base qui fait arriver en retard, ou sans objet ni balise, à un objectif.",
        "Fais ton retour 60 à 90 s avant l'objectif, achète une balise de contrôle."),
    ErrorType(
        "COMBAT_MAUVAIS_PLACEMENT", "Mauvais placement en combat", KEY_MOMENT, AI,
        "Engagé en premier ou hors de portée de son équipe, ou cible prioritaire laissée "
        "libre pendant un combat.",
        "Reste à portée de tes alliés et attends l'engagement de ton équipe.", theme="combat"),
    ErrorType(
        "OBJ_COMBAT_EN_INFERIORITE", "Objectif contesté en infériorité", KEY_MOMENT, AI,
        "Objectif contesté avec moins de joueurs ou sans les sorts importants.",
        "Cède l'objectif et prends autre chose ailleurs quand le combat est perdu d'avance."),
)

BY_CODE = {error.code: error for error in CATALOGUE}
assert len(BY_CODE) == len(CATALOGUE), "codes d'erreur en double"


def for_role(role: str) -> list[ErrorType]:
    return [error for error in CATALOGUE if role in error.roles]


def describe_for_ai(role: str | None = None) -> list[dict[str, str]]:
    """Le catalogue sous une forme simple à donner à une IA (codes à choisir)."""
    errors = CATALOGUE if role is None else tuple(for_role(role))
    return [{"code": e.code, "nom": e.label, "description": e.description,
             "detection": e.detected_by} for e in errors]
