# Fiches de connaissances

Ce dossier contient ce que le coach « sait » : une fiche Markdown par concept, rangée par rôle
(`jungle/`, `top/`…). Le coach n'utilise que les fiches **validées**.

```markdown
---
id: jgl-setup-drake
titre: Préparer un dragon
role: JUNGLE            # TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY ou TOUS
theme: objectifs        # pathing, objectifs, ganks, tracking, vision, invade, farm, combat, macro, mental, general
champions: []           # vide = valable pour tous les champions
patch_min: "16.20"
source: youtube:abc123
statut: a_valider       # a_valider, valide ou obsolete
---
Être sur place 45 s avant l'apparition, avec la priorité de la botlane…
```

Créer des fiches :

```bash
lolcoach-fiches video "https://www.youtube.com/watch?v=…" --role JUNGLE   # API Gemini
lolcoach-fiches consigne --role JUNGLE      # ou : coller la consigne dans l'appli Gemini,
lolcoach-fiches importer reponse.json       #      puis importer sa réponse
lolcoach-fiches lister --statut a_valider   # relire…
lolcoach-fiches valider jgl-setup-drake     # …et valider
```

Pour ne pas relire : `LOLCOACH_FICHES_AUTO_VALIDER=1` dans `.env`. Les fiches créées par
`lolcoach-fiches` ou par le relais arrivent alors directement « valide ».

### Fiches privées

Ce dépôt est public. Les fiches tirées d'une formation payante ou privée vont dans
`connaissances/prive/` (mêmes sous-dossiers par rôle) : le coach les lit comme les autres, mais
Git les ignore, elles ne partent jamais sur GitHub. N'y mets que des fiches rédigées avec tes
mots, jamais les transcriptions ni les vidéos elles-mêmes.

### Depuis une session Claude du cloud : le relais du PC

Une session du cloud n'a ni `agy` ni ton compte Google : elle confie la vidéo à ton PC par
GitHub.

1. **Sur le PC**, une fois : `LOLCOACH_FICHES=agy` dans `.env`, puis double-clic sur
   `outils\relais_lolcoach.cmd` (ou `lolcoach-relais surveiller`). Il regarde toutes les 20 s
   les branches `relais-fiches-*`, une demande à la fois, et ne touche ni à ta branche ni à
   tes fichiers. Tant qu'il tourne, le PC ne se met pas en veille.
2. **Dans le cloud** : `lolcoach-relais demander "https://www.youtube.com/watch?v=…" --role JUNGLE`,
   puis `lolcoach-relais lire` quelques minutes plus tard (code 3 = encore en attente). Les
   fiches arrivent « à valider » dans ce dossier.

Ce qui voyage : l'identifiant de la vidéo et le rôle, rien d'autre. Le PC reconstruit lui-même
le lien et lance toujours sa propre commande. Une session du cloud peut pousser une branche mais
pas la supprimer : elle marque la branche comme lue, et le PC la supprime au passage suivant.
Pour un autre préfixe de branche (si le dépôt refuse `relais-*`), `LOLCOACH_RELAIS_PREFIXE`
des deux côtés.

Une fiche dépassée par un patch passe en `obsolete` (`lolcoach-fiches obsolete <id>`) au lieu
d'être supprimée : Git garde l'historique.
