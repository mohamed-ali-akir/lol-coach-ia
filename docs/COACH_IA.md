# Le coach IA : installation et utilisation

Ce guide met en place tout ce qui fait de LoL Coach IA un coach « comme un humain » : rapport
détaillé après chaque partie, captures du replay regardées par une IA, suivi des erreurs d'une
partie à l'autre, fiches tirées de vidéos éducatives.

Tout tourne **sur ton PC**, celui où League of Legends est installé (Windows ou macOS).

## 1. Installation

```bash
python -m venv .venv
.venv\Scripts\activate                 # macOS : source .venv/bin/activate
pip install -e ".[mcp,gemini]"         # mcp : Claude Desktop / Claude Code ; gemini : API Gemini
cp .env.example .env                   # puis remplis-le (voir ci-dessous)
```

Dans `.env`, au minimum :

```ini
RIOT_API_KEY=RGAPI-…                   # developer.riotgames.com
RIOT_PLATFORM=euw1
LOLCOACH_RIOT_ID=TonPseudo#EUW
```

## 2. Les références « meilleurs joueurs »

Le coach compare ta partie à ce que font les joueurs MASTER, GRANDMASTER et CHALLENGER **sur ton
champion** (or, farm, monstres et zone occupée minute par minute).

```bash
lolcoach-collecte classement CHALLENGER                   # ~300 joueurs par serveur
lolcoach-collecte classement GRANDMASTER
lolcoach-collecte classement MASTER --max 2000            # les 2 000 meilleurs Master
lolcoach-collecte parties --joueurs 500 --par-joueur 20   # ~1 400 parties/heure (clé de dev)
lolcoach-collecte joueur "TonPseudo#EUW" --parties 30
lolcoach-entrainer                                        # repères, victoire, références
```

Une référence par champion demande au moins 20 parties de ce champion à ce rôle : plus la
collecte est longue, plus il y a de champions couverts (`lolcoach-entrainer` affiche les junglers
les mieux documentés). Les parties collectées avant cette version n'ont pas le farm de jungle
séparé : `lolcoach-collecte reimporter` le reconstitue depuis l'archive, sans rien retélécharger.

## 3. Le coach dans Claude (ton abonnement, sans clé d'API)

Le serveur MCP `lolcoach-mcp` donne à Claude des outils pour lire tes parties, tes erreurs et les
fiches, puis enregistrer son rapport.

**Claude Desktop** : ajoute dans `claude_desktop_config.json`
(Windows : `%APPDATA%\Claude\claude_desktop_config.json`, macOS :
`~/Library/Application Support/Claude/claude_desktop_config.json`) :

```json
{
  "mcpServers": {
    "lol-coach": {
      "command": "C:\\chemin\\vers\\lol-coach-ia\\.venv\\Scripts\\lolcoach-mcp.exe",
      "env": {
        "LOLCOACH_RIOT_ID": "TonPseudo#EUW",
        "DATABASE_URL": "sqlite:///C:/chemin/vers/lol-coach-ia/donnees/lolcoach.db",
        "LOLCOACH_MODELS_DIR": "C:/chemin/vers/lol-coach-ia/modeles",
        "LOLCOACH_KNOWLEDGE_DIR": "C:/chemin/vers/lol-coach-ia/connaissances"
      }
    }
  }
}
```

Quitte complètement Claude Desktop et relance-le, puis demande : « Analyse ma dernière partie ».

**Claude Code** : le fichier `.mcp.json` du projet déclare déjà le serveur. Lance Claude Code
depuis le dossier du projet, environnement virtuel activé. Le skill `/analyse-game` (dossier
`.claude/skills/`) déroule toute l'analyse : `/analyse-game` ou `/analyse-game EUW1_7123456789`.
Hors du projet : `claude mcp add lol-coach -- /chemin/vers/.venv/bin/lolcoach-mcp`.

Usage personnel seulement : un abonnement Claude ne se partage pas avec d'autres joueurs. Pour
une application ouverte à d'autres, il faut l'API (clé ANTHROPIC_API_KEY).

## 4. Le rapport automatique après chaque partie

```bash
lolcoach-auto activer-replay-api     # une fois : EnableReplayApi=1 dans game.cfg (sauvegarde .bak)
lolcoach-auto verifier               # client, Replay API, IA : tout est-il prêt ?
lolcoach-auto surveiller --replay    # laisse tourner pendant que tu joues
```

À chaque fin de partie classée :

1. la partie est récupérée chez Riot (elle y apparaît quelques minutes après) ;
2. les erreurs du catalogue sont repérées dans la timeline, les moments clés choisis (morts les
   plus coûteuses, objectifs perdus, autres erreurs datées ; 6 au plus) ;
3. avec `--replay` : le replay est téléchargé et ouvert, chaque moment est enregistré en images
   (1 par seconde, de 20 s avant à 8 s après, caméra sur ton champion, brouillard de guerre de
   ton équipe), puis le jeu est refermé. Ne touche pas au clavier pendant ce temps ;
4. l'IA de vision regarde les images et relève les erreurs (codes du catalogue seulement) ;
5. l'IA de rédaction écrit le rapport : `donnees/rapports/<partie>.md`, et en base.

Sans IA configurée, tu obtiens quand même le rapport factuel complet. Pour une partie déjà
jouée : `lolcoach-auto partie` (la dernière) ou `lolcoach-auto partie --partie EUW1_… --replay`.

Un replay ne se lit que sur le patch où la partie a été jouée : analyse tes parties le jour même.

### Choisir les IA

| Variable | Valeurs | Remarques |
|---|---|---|
| `LOLCOACH_VISION` | `aucun`, `gemini`, `agy`, `claude-code`, `commande` | regarde les captures du replay |
| `LOLCOACH_REDACTEUR` | `aucun`, `gemini`, `agy`, `claude-code`, `claude`, `commande` | écrit le rapport |

- `gemini` : API Gemini, modèle `gemini-4-argon` par défaut (`LOLCOACH_VISION_MODEL`,
  `LOLCOACH_REDACTEUR_MODEL`), clé `GEMINI_API_KEY`. Payant à l'usage, sans offre gratuite pour
  Argon d'après la documentation de Google.
- `agy` : Gemini par la ligne de commande Google Antigravity (`agy -p`), avec ton compte : c'est
  ton relais. La commande lancée est `agy -p <consigne> --output-format json --json-schema
  <fichier>` ; pour la changer (modèle, options de permission), définis `LOLCOACH_VISION_CMD`
  ou `LOLCOACH_REDACTEUR_CMD`. En mode non interactif, `agy` refuse les outils non autorisés :
  s'il ne lit pas les images, autorise la lecture de fichiers dans ses réglages.
- `claude-code` : `claude -p … --allowedTools Read`, avec ton abonnement Claude.
- `claude` : API Claude (`ANTHROPIC_API_KEY`), rédaction seulement.
- `commande` : n'importe quel programme. Dans `LOLCOACH_VISION_CMD` / `LOLCOACH_REDACTEUR_CMD`,
  ces jetons sont remplacés : `{prompt}` (la consigne), `{prompt_file}`, `{schema_file}` (le
  schéma JSON attendu), `{images}` (un argument par image), `{dossier}`. Rien ne passe par un
  interpréteur de commandes ; une consigne de plus de 6 000 caractères est passée par fichier.

Exemple, vision par Gemini (relais `agy`), rédaction par Claude (abonnement) :

```ini
LOLCOACH_VISION=agy
LOLCOACH_REDACTEUR=claude-code
```

### Garde-fous

- L'IA de vision ne peut enregistrer que des **codes du catalogue** (`lolcoach-coach catalogue`),
  avec une minute valide et une confiance entre 0 et 1 ; le reste est écarté et listé. Une erreur
  qu'elle croit nouvelle devient une **proposition** (`lolcoach-coach erreurs PUUID
  --propositions`), jamais une erreur.
- Un rapport rédigé qui cite des nombres absents des données porte la mention « À vérifier ».
- Tu peux contester une erreur : `lolcoach-coach erreurs PUUID --contester 12`. Elle ne compte
  plus dans ta progression.

## 5. Le suivi de ton niveau

```bash
lolcoach-coach analyser PUUID              # repère les erreurs dans tes 20 dernières parties
lolcoach-coach erreurs PUUID               # progression
lolcoach-coach erreurs PUUID --partie EUW1_…
lolcoach-coach note PUUID "Renekton à 2 PV, j'ai tenté" --partie EUW1_… --minute 8:10
lolcoach-coach catalogue
```

Chaque erreur a un code fixe et appartient à un type de situation (partie, monstre épique,
moment analysé en vidéo). Le taux d'erreur compare les 5 dernières parties aux 10 précédentes :
🔁 récurrente (3 parties sur 5), 📉 en hausse, 🆕 nouvelle, • occasionnelle, 📈 en progrès,
✅ corrigée (plus vue sur les dernières situations après au moins 2 fois avant). Le rapport
d'après-partie affiche ce suivi, et le coach IA s'en sert pour te dire ce que tu refais et
ce que tu as corrigé.

## 6. Les fiches de connaissances

Voir [`connaissances/README.md`](../connaissances/README.md). En résumé : une vidéo YouTube
devient jusqu'à 8 fiches « à valider » (`lolcoach-fiches video <lien>`), tu les relis, tu valides
celles qui sont justes. Mieux vaut 50 vidéos récentes et bien choisies que 500 transcriptions :
le coach n'utilise que des fiches validées.

## 7. Ce que l'outil ne fait pas

- Pas de conseil **pendant** la partie : les règles de Riot l'interdisent pour toute
  information que le jeu ne montre pas. Tout est analysé après la partie.
- Le fichier `.rofl` n'est pas décodé : le replay est lu par le jeu lui-même, via la Replay API.
- La position des joueurs n'est connue qu'une fois par minute dans les données de Riot ; les
  captures du replay complètent sur les moments clés.
