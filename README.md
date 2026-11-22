# LoL Coach IA

Un coach pour League of Legends qui repose sur **des modèles entraînés maison** à partir de vraies parties classées, et pas sur un simple chatbot.

Il fait ce que fait un coach :

1. **Il diagnostique** : tes statistiques comparées à celles des joueurs de ton rang et de ton rôle, et **3 priorités au plus**, une par thème (farm, phase de lane, survie, vision, jeu en équipe, combat). Un retard ne devient une priorité que s'il est net et qu'il ne peut guère venir du hasard : un joueur dans la moyenne n'en reçoit en général aucune.
2. **Il montre les moments clés** : chaque mort est classée (prise seul, gank, duel perdu…) et chiffrée en points de probabilité de victoire, **à la minute près**.
3. **Il donne un conseil et un exercice** pour chaque point faible.
4. **Il fixe des objectifs** (« 5,3 de farm par minute sur tes 10 prochaines parties ») et vérifie s'ils sont tenus.

Exemple de sortie (`lolcoach-coach rapport`) :

```
Partie du 13/06/2026 : Ahri (mid), défaite en 29:12, 1/7/3.

Le fil de la partie
Ton équipe a eu jusqu'à 86 % de chances de gagner (à 16:40) avant de laisser filer la partie.

Les moments qui ont coûté le plus cher
- 6:28, duel perdu : 1 adversaire sur toi, 3 alliés à portée. Les chances de victoire de ton équipe
  passent de 55 % à 46 % (coût estimé : 9 points). L'adversaire prend un dragon dans la foulée.
  À retenir : Note ce qui a fait perdre le duel (niveau, objets, sorts d'invocateur en recharge)…
```

## Essayer sans clé Riot

```bash
python -m venv .venv
source .venv/bin/activate              # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
lolcoach-demo                          # simule 600 parties, entraîne les modèles, affiche diagnostic et rapport
```

La démo utilise des **parties simulées** : elle montre le fonctionnement, pas la qualité réelle des modèles.

## Utilisation avec tes vraies parties

```bash
cp .env.example .env                   # puis colle ta clé Riot dedans (developer.riotgames.com)

# 0. Vérifier que tout fonctionne avec ta clé (une partie, rien n'est enregistré)
lolcoach-collecte verifier "TonPseudo#EUW"

# 1. Constituer la base de comparaison : des joueurs de ton rang et leurs parties
lolcoach-collecte classement GOLD II
lolcoach-collecte parties --joueurs 50 --par-joueur 20 --boule-de-neige

# 2. Tes propres parties (affiche ton identifiant « puuid », à réutiliser ensuite)
lolcoach-collecte joueur "TonPseudo#EUW" --parties 30

# 3. Entraîner les modèles
lolcoach-entrainer

# 4. Le coach
lolcoach-coach diagnostic PUUID
lolcoach-coach rapport PUUID                       # dernière partie
lolcoach-coach objectifs PUUID --proposer
lolcoach-coach objectifs PUUID --creer cs_per_min
```

Avec une clé de développement (100 requêtes toutes les 2 minutes, 2 requêtes par partie), on récupère environ **1 400 parties par heure**. La collecte s'interrompt sans perte (Ctrl+C), ne retélécharge jamais une partie connue, et retente toute seule à la collecte suivante une partie dont le téléchargement a échoué.

**Si tu changes de type de clé Riot** (clé de développement → clé personnelle ou de production), les identifiants de joueurs déjà en file deviennent invalides : Riot les calcule par application. Les parties déjà collectées restent utilisables ; relance simplement `classement` et `joueur` pour reconstituer la file.

`verifier` teste toute la chaîne (compte, rang, liste des parties, partie, timeline, analyse) et dit ce qui passe ou non. Il liste aussi les types d'événements que Riot envoie et que le projet ignore : c'est la première chose à relancer si un jour la collecte se met à échouer après un changement de l'API.

Autres commandes de collecte : `completer` (timelines manquantes), `reimporter` (reconstruit la base depuis l'archive, sans contacter Riot), `parties --rafraichir-apres 24` (reprend les joueurs déjà vus).

## Dans LoL Companion

L'onglet **Coach** de [LoL Companion](https://github.com/mohamed-ali-akir/lol-companion) affiche le diagnostic, les rapports et les objectifs. Il parle à ce projet par une API locale :

```bash
pip install -e ".[api]"
lolcoach-api                           # écoute sur http://127.0.0.1:8765
```

Au premier lancement, un jeton d'accès est écrit dans `donnees/api-token` : colle-le dans l'onglet Coach.

## Réécriture par un modèle de langage (facultatif)

Tout fonctionne sans : le texte de base est complet et gratuit. Avec `--ia`, un modèle de langage le réécrit de façon plus fluide.

| `LOLCOACH_LLM` | Coût | Remarques |
|---|---|---|
| `aucun` (défaut) | gratuit | texte de base |
| `ollama` | gratuit | modèle local, [Ollama](https://ollama.com) doit tourner sur ton PC |
| `claude` | payant | `pip install -e ".[claude]"` et `ANTHROPIC_API_KEY` ; modèle par défaut `claude-opus-5-5`, `LOLCOACH_LLM_MODEL=claude-haiku-5-5` coûte bien moins cher |

Garde-fou : le modèle ne reçoit que le texte factuel, et sa réponse est rejetée si elle contient **un seul nombre ou une seule minute absents des faits** (signe compris), des chiffres regroupés autrement, ou une évolution « de X % à Y % » inversée. En cas de rejet ou de panne, tu obtiens le texte de base. Limite : un nombre écrit en toutes lettres n'est pas vérifié, c'est pourquoi le texte de base reste la référence.

## Ce que les modèles savent faire, et leurs limites

| Modèle | Rôle | Comment il est évalué |
|---|---|---|
| Probabilité de victoire | chances de l'équipe à chaque minute, à partir de l'écart d'or, d'XP, des tours, dragons, etc. | sur des parties plus récentes que celles d'entraînement : précision, score de Brier, **calibration** |
| Repères | répartition de chaque statistique par rang, rôle et résultat | nombre de parties par groupe ; repli sur « tous rangs » si un rang manque de données |
| Draft | victoire selon les 10 champions | face à une prédiction naïve ; attends 52 à 56 % sur de vraies parties |

À savoir avant de s'y fier :

- **La position des joueurs n'est connue qu'environ une fois par minute.** « Pris seul » veut dire « probablement seul ».
- **Le coût d'une mort est une estimation** : la baisse de probabilité de victoire autour de la mort, partagée entre les coéquipiers morts au même moment.
- **Le rang d'une partie est approximatif** : c'est celui du joueur par lequel elle a été trouvée.
- L'API de Riot ne dit rien des sorts ratés, du placement en combat ni de l'état des vagues. Un coach humain qui regarde la vidéo voit plus.
- Il faut **au moins quelques centaines de parties** de ton rang pour des repères fiables, et 5 de tes parties à un rôle pour un diagnostic (30 pour une fiabilité « bonne »).

### Ce qui a été vérifié, et ce qui ne l'a pas encore été

| | État |
|---|---|
| Collecte, analyse, modèles, coach, API | vérifiés par les tests, sur SQLite et MariaDB, avec des **parties simulées** |
| Tolérance au format réel de Riot (champs et événements en plus, cas limites documentés) | vérifiée par `tests/test_format_reel.py`, sur des réponses **reconstruites d'après la documentation** |
| Connexion au vrai serveur de Riot et lecture de ses refus | vérifiée (clé invalide → message clair, rien n'est enregistré) |
| **Lecture de vraies parties avec une vraie clé** | **pas encore faite** : c'est le rôle de `lolcoach-collecte verifier` |
| **Qualité des modèles sur de vraies parties** | **inconnue** : les chiffres de la démo viennent de parties simulées et ne disent rien du jeu réel. `lolcoach-entrainer` affiche les vraies mesures après une collecte |

## Sécurité et vie privée

- La clé Riot et la clé du modèle de langage restent dans `.env`, jamais dans Git ni dans les journaux.
- La base ne contient **aucun pseudo** : les joueurs sont désignés par l'identifiant opaque de Riot (puuid).
- L'API n'écoute que sur ton PC, exige un jeton sur toutes ses routes et le vérifie **avant de lire le corps de la requête** (limité à 16 Ko), vérifie l'en-tête `Host`, n'autorise aucun site web à lire ses réponses, et plafonne les réécritures payantes par heure.
- Les modèles sont des fichiers **JSON** : pas de fichier binaire exécutable à l'ouverture (pickle).
- Tout ce qui vient de l'extérieur (réponses de Riot, du modèle de langage, requêtes HTTP) est validé avant usage.

Ce projet n'analyse que des parties **terminées**. Un produit public doit être enregistré auprès de Riot et respecter ses [règles développeur](https://developer.riotgames.com/policies/general). LoL Coach IA n'est ni approuvé ni soutenu par Riot Games.

## Développement

```bash
ruff check .          # style
mypy                  # types, mode strict
pytest -q             # 193 tests, sans réseau (réponses Riot simulées)
python benchmarks/mesures.py   # temps de chaque étape, pour vérifier qu'une modification ne ralentit rien

# Les mêmes tests sur MySQL / MariaDB :
LOLCOACH_TEST_DATABASE_URL=mysql+pymysql://utilisateur:motdepasse@localhost/ pytest -q
```

La CI lance tout cela sur Python 3.11 et 3.13, sur MariaDB, avec les versions figées de `requirements-lock.txt`, et vérifie que le paquet installé fait tourner la démo.

Le fonctionnement interne, les choix de conception et la marche à suivre pour faire évoluer le projet (nouvelle statistique, migration de la base, changement de l'API Riot) sont dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Les mesures de performance, les optimisations faites et celles volontairement écartées sont dans [docs/PERFORMANCES.md](docs/PERFORMANCES.md).

Ordres de grandeur (1 500 parties simulées) : entraînement complet en 3 secondes, diagnostic en 34 ms au premier calcul puis moins d'une milliseconde, rapport d'une partie en 3 ms.

## Feuille de route

- [x] Collecte des parties et des timelines (API Riot), archive brute
- [x] Diagnostic par rang et par rôle, 3 priorités
- [x] Probabilité de victoire et coût de chaque mort
- [x] Rapport d'après-partie avec les minutes citées
- [x] Objectifs suivis, API, onglet Coach dans LoL Companion
- [ ] Rôles, duos et contres dans le modèle de draft
- [ ] Conseils avant la partie, pendant la sélection des champions
- [ ] Analyse du mental : résultats après deux défaites, selon l'heure, selon la longueur de la session
- [ ] Faire noter des rapports à l'aveugle par un joueur de haut niveau
