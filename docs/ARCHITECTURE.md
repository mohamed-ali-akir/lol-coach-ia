# Architecture et choix de conception

Ce document explique comment le projet est construit et pourquoi, pour qu'on puisse le faire évoluer dans plusieurs années sans tout relire.

## Vue d'ensemble

```
API Riot ──► collecte ──► base (SQLite / MySQL) ──► analyse ──► coach ──► ligne de commande
                │                                      ▲           │
                └──► archive brute (JSON compressé)    │           └──► API web ──► LoL Companion
                                                    modèles (JSON)
```

| Dossier | Rôle |
|---|---|
| `riot.py` | Seul fichier qui connaît les adresses de l'API Riot : limites de débit, nouvelles tentatives, erreurs |
| `collecte/` | Réponses de Riot → lignes de la base (`parse.py`, `timeline.py`), enchaînement (`pipeline.py`) |
| `archive.py` | Copie compressée des réponses d'origine |
| `db.py`, `migrations/` | Schéma et son évolution |
| `analyse/data.py` | Seul fichier qui lit les parties en base : tout le reste part de `MatchData` |
| `analyse/` | Statistiques, repères, diagnostic, morts, rapport (les faits) |
| `modele/` | Modèles entraînés : régression logistique, draft, probabilité de victoire |
| `coach/` | Mise en phrases, modèle de langage facultatif, objectifs, service commun |
| `api/` | API web locale |
| `simulation.py`, `demo.py` | Parties simulées au format Riot, pour les tests et la démo |

## Les choix qui comptent

### 1. Tout ce qui vient de l'extérieur est suspect

Les réponses de Riot, celles du modèle de langage et les requêtes HTTP sont validées à l'entrée :

- `riot.py` vérifie la forme des réponses (objet ou liste) et valide tout identifiant avant de le mettre dans une adresse ;
- `parse.py` et `timeline.py` écartent une réponse au format inattendu au lieu de planter : une collecte de plusieurs heures ne s'arrête pas pour une partie abîmée ;
- les types d'événements ou de monstres inconnus sont ignorés ou stockés tels quels : un monstre ajouté au jeu plus tard ne casse rien ;
- `archive.py` valide les noms avant d'en faire un chemin de fichier.

### 2. On peut toujours tout recalculer

Riot efface les timelines au bout d'un an environ et les parties au bout de deux. L'archive garde les réponses d'origine : si le schéma change ou si une statistique est ajoutée, `lolcoach-collecte reimporter` reconstruit la base sans rien retélécharger. Désactiver l'archive (`LOLCOACH_ARCHIVE_DIR` vide) fait gagner de la place, mais on perd cette garantie.

### 3. Le schéma n'évolue que par migrations

`db.init_db` applique les migrations Alembic au démarrage. Sous SQLite, une migration est atomique : en cas d'échec, la base reste intacte (testé). Une base créée par la version 0.1, antérieure aux migrations, est reconnue et mise à jour sans perte.

Règles :

- ne jamais modifier une migration publiée, en ajouter une nouvelle ;
- un test compare le schéma obtenu par migrations au schéma déclaré dans `db.py` : oublier une migration fait échouer les tests ;
- pas de `ON DELETE CASCADE` : les suppressions sont explicites (`delete_match`), ce qui garde les migrations simples et portables.

### 4. Les modèles sont du JSON

Un fichier `pickle` exécute du code à l'ouverture et n'est relisible qu'avec la même version de scikit-learn. Ici, scikit-learn ne sert qu'à l'entraînement ; un modèle enregistré est une liste de nombres, relue et appliquée avec numpy seul (`modele/logistic.py`). Chaque fichier porte un numéro de format : une version future du projet refusera proprement un format qu'elle ne connaît pas.

### 5. Les faits d'abord, le texte ensuite

`analyse/report.py` ne produit que des données vérifiables. `coach/redaction.py` les met en phrases, de façon entièrement déterminée : mêmes données, même texte. Un modèle de langage peut réécrire ce texte (`coach/llm.py`), sous trois garde-fous :

1. il ne reçoit que le texte factuel, sans pseudo ni donnée brute ; le nom du champion est nettoyé ;
2. sa réponse est rejetée si elle contient un chiffre ou une minute absents du texte factuel, un lien, ou une longueur anormale ;
3. toute panne, tout refus ou tout rejet rend le texte factuel.

Le projet ne dépend donc jamais d'un modèle de langage pour fonctionner.

### 6. L'heure réelle des photos, jamais leur numéro

Depuis le patch 16.1, les photos de la timeline sont espacées de 60,5 à 71 secondes. On stocke l'heure de chaque photo et on interpole : « à 15 minutes » veut dire « à 900 000 ms », pas « à la 15e photo ».

### 7. Préférer les données aux constantes du jeu

Les règles du jeu changent tous les ans. Plutôt que de coder l'heure d'apparition d'un dragon, on regarde ce qui s'est passé : « l'adversaire a pris un objectif dans les 90 secondes après ta mort ». Les rares constantes de jeu restantes sont regroupées et commentées (durée d'effet du Baron dans `modele/winprob.py`, seuils du classement des morts dans `analyse/deaths.py`).

## Comment les analyses fonctionnent

### Repères et diagnostic

Pour chaque statistique, on garde ses quantiles par groupe (rang, rôle, victoire ou défaite). Victoires et défaites sont séparées, parce qu'un joueur qui perd beaucoup a mécaniquement de moins bonnes statistiques : on compare ses victoires aux victoires des autres, ses défaites à leurs défaites.

Le poids d'une statistique est sa capacité, **prise seule**, à distinguer gagnants et perdants à ce rôle (AUC). Un modèle unique avec toutes les statistiques donnait presque tout le poids à une seule de deux statistiques liées (farm et or), ce qui masquait les vraies causes.

Priorité = retard du joueur × poids. Le diagnostic garde au plus une priorité par thème, pour ne pas répéter trois fois le même conseil.

### Probabilité de victoire

Une régression logistique par tranche de 5 minutes, sur l'état de la partie vu des bleus (écarts d'or, d'XP, de kills, de tours, dragons, âme, Baron et Dragon ancestral récents, autres monstres). Entre deux tranches, les prédictions sont mélangées progressivement, pour éviter un saut artificiel qui fausserait le coût d'une mort survenue à cet instant.

L'évaluation découpe **par partie** et **dans l'ordre chronologique** : toutes les photos d'une partie sont du même côté. On mesure la calibration (quand le modèle annonce 70 %, l'équipe gagne-t-elle 7 fois sur 10 ?), car c'est la probabilité elle-même qui sert.

### Coût d'une mort

Probabilité de victoire de l'équipe à la dernière photo avant la mort, moins celle 2 minutes après. La baisse est partagée à parts égales entre les coéquipiers morts dans l'intervalle ; une mort dans un combat gagné coûte 0. C'est une estimation utile pour classer les erreurs, pas une mesure causale.

### Classement des morts

Règles simples, appliquées dans l'ordre (voir `classify_deaths`). La distance de 2 000 unités pour « un allié est à côté » est celle des statistiques e-sport de Riot. Les positions étant interpolées entre deux photos, « pris seul » signifie « probablement seul ».

## Faire évoluer le projet

**Ajouter une statistique.** Une entrée dans `METRICS` et son thème dans `FAMILY_OF` (`analyse/metrics.py`), puis `lolcoach-entrainer` pour recalculer les repères. Diagnostic, rapport, objectifs et API la prennent en compte sans autre changement.

**Changer le schéma.** Modifier `db.py`, ajouter un fichier dans `migrations/versions/` (le suivant de la série), lancer les tests. Si la donnée vient de Riot, adapter `parse.py` ou `timeline.py` ; l'archive permet de remplir la nouvelle colonne pour les parties déjà collectées (`reimporter`).

**Riot change son API.** Les adresses sont dans `riot.py`, la lecture des réponses dans `collecte/parse.py` et `collecte/timeline.py`. Le simulateur (`simulation.py`) produit le même format : l'adapter met à jour tous les tests d'un coup.

**Changer de modèle.** Tout modèle exportable en nombres (coefficients) entre dans le format JSON actuel. Pour un modèle plus complexe, écrire son propre format JSON versionné plutôt que d'introduire un fichier binaire.

**Changer le format d'un fichier de modèle.** Augmenter `FORMAT_VERSION` : les anciens fichiers sont alors refusés avec un message qui demande de relancer l'entraînement.

**Mettre à jour les dépendances.** Dependabot propose les mises à jour chaque semaine. La CI teste à la fois les dernières versions autorisées et les versions figées de `requirements-lock.txt` ; pour régénérer ce fichier après une mise à jour : `pip freeze --exclude-editable > requirements-lock.txt` dans un environnement propre où les tests passent.

## Sécurité de l'API

Même en local, une API sans protection est exposée : n'importe quelle page web ouverte dans le navigateur peut envoyer des requêtes à `http://127.0.0.1`. D'où :

- un jeton exigé sur toutes les routes sauf `/health`, comparé en temps constant ; s'il n'est pas fourni, il est généré et rangé dans un fichier lisible par l'utilisateur seul ;
- la vérification de l'en-tête `Host` (attaque par « DNS rebinding ») ;
- aucune autorisation CORS ;
- des entrées validées par des formats stricts, des corps de requête qui refusent les champs inconnus ;
- un plafond horaire de réécritures par le modèle de langage, et une seule synchronisation Riot à la fois ;
- un refus de démarrer en écoute hors du PC sans réglage explicite et sans jeton fort ;
- des erreurs internes sans détail technique.

## Tests

Aucun test ne contacte le réseau. Les réponses de Riot sont produites par le simulateur ou construites à la main ; le modèle de langage est remplacé par un faux. Une base de démonstration de 300 parties, avec un joueur fictif aux défauts connus, sert de référence : le diagnostic doit retrouver ces défauts.

Les tests tournent sur SQLite et sur MariaDB (`LOLCOACH_TEST_DATABASE_URL`). Passer sur MariaDB a par exemple révélé que les colonnes décimales étaient en simple précision sur MySQL.
