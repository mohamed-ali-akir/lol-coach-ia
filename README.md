# LoL Coach IA

Un coach pour League of Legends qui repose sur **un modèle d'IA entraîné maison** à partir de vraies parties classées, et pas sur un simple chatbot.

- **Collecte** : récupère des milliers de parties classées via l'API Riot (gratuite), en respectant les limites de débit.
- **Modèle** : régression logistique entraînée sur les drafts, qui prédit l'équipe gagnante et mesure la force de chaque champion dans le patch.
- **Bientôt** : analyse de tes points faibles par rapport aux joueurs de ton rang, conseils rédigés par un modèle de langage, API pour l'onglet « Coach » de LoL Companion.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -e ".[dev]"            # ajoute ",mysql" pour MySQL / MariaDB
cp .env.example .env               # puis colle ta clé Riot dedans
```

Clé Riot : [developer.riotgames.com](https://developer.riotgames.com). La clé de développement expire toutes les 24 h.

## Utilisation

```bash
# 1. Trouver des joueurs : par rang, ou à partir de ton propre compte
lolcoach-collecte classement GOLD II
lolcoach-collecte joueur "TonPseudo#EUW"

# 2. Télécharger leurs parties classées (Ctrl+C sans risque : chaque partie est enregistrée au fil de l'eau)
lolcoach-collecte parties --joueurs 20 --par-joueur 20 --boule-de-neige

# 3. Entraîner et évaluer le modèle
lolcoach-entrainer
lolcoach-entrainer --patch X.Y     # un seul patch (2 premiers nombres de gameVersion)
```

Avec une clé de développement (100 requêtes / 2 min), on récupère jusqu'à **2 800 parties environ par heure** (une requête par partie). La base est SQLite par défaut, et MySQL / MariaDB se règle via `DATABASE_URL`.

## Comment le modèle fonctionne

Chaque partie devient un vecteur avec une case par champion : **+1** s'il est chez les bleus, **−1** chez les rouges, 0 sinon. La régression logistique apprend un poids par champion (sa force) et un biais (l'avantage du côté bleu).

Le découpage entraînement / test est **chronologique** : on apprend sur les parties anciennes et on évalue sur les plus récentes. Mélanger les deux ferait « tricher » le modèle en lui montrant le futur.

Chaque évaluation est comparée à une **référence naïve** (toujours prédire le taux de victoire du côté bleu). Un modèle qui ne bat pas cette référence n'a rien appris.

> **À quoi s'attendre :** prédire une partie avec la draft seule est difficile, car le niveau des joueurs compte beaucoup plus que les champions. Sur de vraies données, une précision de **52 à 56 %** est un résultat honnête. Les 82 % visibles dans les tests viennent de données simulées où les champions ont un effet volontairement fort. Ce qui compte, c'est l'écart avec la référence et le classement des champions.

## Structure

```
src/lolcoach/
├─ config.py           lecture du .env, routage Riot (euw1 → europe)
├─ riot.py             client API Riot : limites de débit, gestion des 429
├─ db.py               schéma (players, matches, participants)
├─ collecte/
│  ├─ parse.py         réponse match-v5 → lignes ; écarte remakes et rôles manquants
│  ├─ pipeline.py      classement → joueurs → parties (+ boule de neige)
│  └─ cli.py           commande lolcoach-collecte
└─ modele/
   ├─ draft.py         modèle de draft : encodage, entraînement, évaluation
   └─ cli.py           commande lolcoach-entrainer
tests/                 sans réseau : réponses Riot simulées, données synthétiques
```

## Feuille de route

- [x] Collecte des parties classées (API Riot)
- [x] Modèle n°1 : prédiction de victoire par la draft, évalué face à une référence naïve
- [ ] Modèle n°2 : points faibles d'un joueur (CS/min, morts, vision) comparés aux gagnants de son rang et de son rôle
- [ ] Conseils en langage naturel (Ollama en local, ou API Claude)
- [ ] API (FastAPI) consommée par l'onglet « Coach » de LoL Companion
- [ ] Rôles et synergies dans le modèle de draft, comparaison avec un gradient boosting

## Règles Riot

Ce projet n'analyse que des parties **terminées**. Toute aide en direct pendant une partie doit respecter les [règles développeur de Riot](https://developer.riotgames.com/policies/general). LoL Coach IA n'est ni approuvé ni soutenu par Riot Games.
