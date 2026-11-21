# Performances : mesures et choix

Ce document dit ce qui a été mesuré, ce qui a été optimisé avec quel gain, et ce qui a été laissé tel quel et pourquoi. « Optimisé » veut dire ici : mesuré, puis amélioré là où le temps part réellement.

## Refaire les mesures

```bash
python benchmarks/mesures.py                  # 1 500 parties simulées
python benchmarks/mesures.py --parties 4000 --json mesures.json
```

Les temps dépendent de la machine. Ce sont les rapports entre deux mesures faites au même endroit qui comptent : mesurer avant une modification, mesurer après, comparer.

## Avant et après (1 500 parties, même machine)

| Opération | Avant | Après | Gain |
|---|---|---|---|
| Chargement des parties pour l'analyse | 463 parties/s | 881 parties/s | × 1,9 |
| Entraînement : repères + probabilité de victoire | 7,5 s | 3,0 s | × 2,5 |
| Mémoire maximale pendant l'entraînement | 235 Mo | 207 Mo | − 12 % |
| Diagnostic d'un joueur, premier calcul | 60 ms | 34 ms | × 1,8 |
| Diagnostic d'un joueur, appels suivants | 60 ms | 0,5 ms | × 120 |
| Rapport d'une partie | 63 ms | 2,9 ms | × 22 |
| Rapport d'une partie, un objectif en cours | 260 ms | 3,4 ms | × 76 |
| Analyse et enregistrement d'une partie | 278 parties/s | 260 parties/s | inchangé (bruit de mesure) |

Les résultats des analyses sont strictement identiques avant et après : `tests/test_performances.py` compare chaque raccourci à la version simple qu'il remplace.

## Ce qui a été fait

1. **Chargement vectorisé.** Les photos (environ 300 lignes par partie) sont converties par blocs avec numpy au lieu d'être traitées ligne par ligne.
2. **Balises non chargées par défaut.** Les poses et destructions de balises forment la majorité des événements et ne servent à aucune analyse actuelle. Elles restent en base, mais ne sont lues que sur demande (`iter_matches(..., wards=True)`).
3. **Recherche triée au lieu de comparaisons deux à deux.** Le classement des morts comparait chaque mort à toutes les autres (1,25 million de comparaisons pour 1 500 parties). Il utilise maintenant une recherche dans la liste triée, calculée une fois par partie et réutilisée pour les 10 joueurs. Même chose pour « objectif perdu après la mort ».
4. **Une seule lecture pour deux modèles.** Les repères et la probabilité de victoire étaient entraînés en lisant deux fois toutes les parties. `train_analysis` les alimente en une seule passe.
5. **Accumulateurs compacts.** L'entraînement ne garde plus les parties en mémoire, seulement des nombres (8 octets par valeur).
6. **Diagnostic gardé en mémoire.** Le rapport, les objectifs et l'onglet Coach le redemandent à chaque affichage alors qu'il ne change qu'après une nouvelle partie. Il est gardé tant que l'empreinte des parties du joueur, son rang et la date des repères sont les mêmes. Le cache est borné (128 entrées).
7. **Objectifs : lecture ciblée.** L'avancement d'un objectif ne relit que les parties jouées depuis sa création, au lieu des 100 dernières.
8. **Masques d'assistances décodés une fois.** Il n'existe que 1 024 masques possibles : ils sont mis en cache.

## Passage à l'échelle

| | 1 500 parties | 4 000 parties |
|---|---|---|
| Entraînement | 3,0 s | 8,1 s |
| Mémoire maximale pendant l'entraînement | 207 Mo | 241 Mo |
| Taille de la base | 58 Mo | 158 Mo |
| Rapport d'une partie | 2,9 ms | 4,0 ms |

Le temps d'entraînement est proportionnel au nombre de parties (environ 2 ms par partie). La mémoire croît de 14 Ko par partie au-dessus d'un socle de 185 Mo (Python, numpy, scikit-learn). Par extrapolation : 100 000 parties demandent environ 3 à 4 minutes d'entraînement, 1,6 Go de mémoire et 4 Go de base. Les temps de réponse du coach ne dépendent pas de la taille de la base, seulement du nombre de parties du joueur.

## Ce qui a été laissé tel quel, et pourquoi

- **L'enregistrement d'une partie (260 par seconde).** Une clé de développement Riot laisse télécharger 0,4 partie par seconde. L'enregistrement est donc environ 600 fois plus rapide que ce qui l'alimente : l'accélérer n'apporterait rien, sauf pour `reimporter` (100 000 parties en 6 minutes, ce qui reste acceptable pour une opération exceptionnelle).
- **Les photos stockées en lignes plutôt qu'en bloc binaire.** Ranger les photos d'une partie dans un seul champ binaire rendrait leur lecture plusieurs fois plus rapide et la base plus petite. Mais on perdrait la possibilité de les interroger en SQL et de les relire sans ce programme, ce qui va contre l'objectif de durée. La lecture représente aujourd'hui 0,5 ms par partie : le gain ne justifie pas ce coût.
- **Un modèle plus lourd pour la probabilité de victoire.** Les modèles à arbres font un peu mieux en fin de partie d'après les publications, mais ils ne s'exportent pas en simple liste de nombres. La régression logistique par tranche de temps reste relisible sans bibliothèque.

## Si un jour cela ne suffit plus

Dans l'ordre du meilleur rapport gain/coût :

1. Ne plus enregistrer les balises (environ un tiers des lignes), si aucune analyse de vision n'est prévue.
2. Entraîner sur les seules parties du patch en cours (`lolcoach-entrainer --patch X.Y`), ce qui est de toute façon plus juste.
3. Passer la base sur MariaDB ou PostgreSQL pour des collectes en parallèle.
4. En dernier recours, stocker les photos en bloc binaire versionné.
