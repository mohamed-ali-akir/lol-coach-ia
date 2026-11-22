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

Une fiche dépassée par un patch passe en `obsolete` (`lolcoach-fiches obsolete <id>`) au lieu
d'être supprimée : Git garde l'historique.
