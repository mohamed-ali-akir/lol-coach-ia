# Qualité sur de vraies parties

Ce document dit ce que valent les modèles et le diagnostic **sur de vraies parties**, et ce que ces données ont permis de corriger. Les chiffres de la démo viennent de parties simulées et ne disent rien du jeu réel ; ceux-ci, si.

## Les données (22/11/2026)

| | |
|---|---|
| Parties | 966 parties classées solo/duo (file 420), serveur EUW, toutes avec leur timeline |
| Joueurs suivis | 68 joueurs Or II tirés du classement officiel, 15 dernières parties chacun |
| Période | patchs 15.5 à 16.23 : certains joueurs jouent peu, leurs 15 dernières parties remontent à un an |
| Clé | clé de développement Riot, 36 minutes de collecte pour 850 parties |
| Repères | un seul rang (Or) : les autres rangs se replient sur « tous rangs confondus » |

Aucune donnée de joueur n'est versionnée : la base et l'archive restent sur la machine qui a collecté.

## La chaîne de collecte

`lolcoach-collecte verifier` passe toutes les étapes sur une vraie partie : compte, rang, liste des parties, partie, timeline, photos (écart de 37 à 60 s, puis de moins d'une seconde pour la dernière), événements. Riot envoie en plus des types d'événements que le projet ignore volontairement : achats, ventes et destructions d'objets, niveaux, sorts, pauses, fin de partie, kills spéciaux.

Sur la collecte complète : statistiques facultatives toutes présentes, aucune partie écartée pour un format inattendu, 4 parties écartées comme remakes ou sans rôles complets.

## Probabilité de victoire

Entraînée sur les 772 parties les plus anciennes, testée sur les 194 plus récentes.

| | Modèle | Référence naïve |
|---|---|---|
| Bonnes prédictions | 72,5 % | 53,3 % |
| Score de Brier (plus bas = mieux) | 0,177 | 0,249 |
| Écart de calibration | 2,8 % | |

| Moment | 0-5 | 5-10 | 10-15 | 15-20 | 20-25 | 25-30 | 30-35 | 35+ min |
|---|---|---|---|---|---|---|---|---|
| Bonnes prédictions | 57,8 % | 69,6 % | 76,4 % | 79,4 % | 78,0 % | 75,5 % | 74,5 % | 67,8 % |

Le modèle est utile et ses probabilités sont fiables : quand il annonce 70 %, l'équipe gagne à peu près 7 fois sur 10. En fin de partie, un seul combat décide souvent de tout, ce qu'aucun écart d'or ne prévoit : la baisse après 35 minutes est attendue.

## Draft

55,2 % de bonnes prédictions contre 52,6 % pour la référence, mais une perte logarithmique moins bonne que la référence. Le modèle n'est donc **pas enregistré**, et son classement des champions n'est pas affiché. Avec 446 parties, il faisait même moins bien que la référence en précision (54,5 % contre 58,0 %). Un modèle de draft demande des dizaines de milliers de parties. Le coach ne s'en sert pas.

Règle ajoutée après cet essai : un modèle qui ne prédit pas mieux que la référence naïve n'est ni enregistré par `lolcoach-entrainer`, ni utilisé par le coach (`useful` dans `modele/winprob.py` et `modele/draft.py`).

## Repères

9 660 performances individuelles. Ce qui distingue le plus les victoires des défaites, à tous les postes : les morts par 10 minutes et le temps passé mort, puis l'écart d'or ou d'XP à 15 minutes face à l'adversaire direct (le farm par minute en jungle). Farm par minute médian : 6,75 en top, 6,49 en jungle, 6,68 en mid, 7,04 en ADC.

## Le diagnostic mesure-t-il quelque chose ?

Test (`benchmarks/stabilite.py`) : les parties de chaque joueur sont coupées en deux moitiés, une partie sur deux. Si une statistique mesure un vrai trait du joueur, les deux moitiés classent les joueurs dans le même ordre.

| Statistique | Joueurs | Accord entre moitiés | Seuil du hasard |
|---|---|---|---|
| Balises de contrôle achetées | 48 | 0,91 | 0,24 |
| Score de vision par minute | 48 | 0,78 | 0,24 |
| Part des dégâts de l'équipe | 31 | 0,77 | 0,30 |
| Farm par minute | 31 | 0,67 | 0,30 |
| Sbires à 10 minutes | 31 | 0,47 | 0,30 |
| Écart d'or à 15 min | 48 | 0,43 | 0,24 |
| Morts par 10 minutes | 48 | 0,41 | 0,24 |
| Temps passé mort | 48 | 0,40 | 0,24 |
| Présence aux objectifs | 47 | 0,38 | 0,24 |
| Morts avant 15 minutes | 48 | 0,37 | 0,24 |
| Participation aux kills | 48 | 0,33 | 0,24 |
| Écart d'XP à 15 min | 48 | 0,32 | 0,24 |
| **Morts seul face à plusieurs adversaires** | 48 | **0,11** | 0,24 |

Douze statistiques sur treize mesurent un trait stable du joueur. La treizième ne se distingue pas du hasard avec 5 à 7 parties par moitié : elle n'est plus jamais une priorité du diagnostic (`priority_eligible=False`), mais reste affichée dans le rapport de chaque partie, où elle décrit un fait.

Sur les 68 joueurs suivis ayant assez de parties à leur poste, 11 n'ont aucune priorité, 28 en ont une, 15 deux, 14 trois. Sur des joueurs simulés tous moyens, 85 % n'en avaient aucune : de vrais joueurs du même rang ont, eux, de vraies forces et faiblesses, ce que confirme le test ci-dessus.

## Défauts trouvés grâce à ces données

| Vu sur de vraies parties | Correction |
|---|---|
| Une collecte lancée juste après une autre commande recevait un 429 : chaque processus ignorait les appels des autres | le limiteur se cale sur le décompte renvoyé par Riot (`X-App-Rate-Limit-Count`) |
| Le modèle de draft, moins bon que la référence, était enregistré et ses « champions les plus forts » affichés | règle « modèle utile » |
| Un jungler recevait « reviens sur une vague de sbires », « ne quitte pas ta lane », « pousse ta vague » | conseils propres au jungler et au support |
| Une partie jouée top recevait l'axe de travail calculé en jungle | axe de travail limité au poste diagnostiqué |
| « Combat d'équipe » avec aucun allié à portée : 3 morts simultanées aux deux bouts de la carte | combat d'équipe = 3 morts à moins de 15 s **et** de 3 000 unités (38 % des morts au lieu de 50 %) |
| « Duel perdu » avec 2 alliés à portée | duel = un adversaire **et** aucun allié à côté |
| « Morts seul » à 0 à côté d'un duel perdu sans allié | libellés précisés : « pris seul à plusieurs » |
| Priorité possible sur une statistique instable | retirée des priorités |

## Limites de cette évaluation

- Un seul rang (Or), un seul serveur (EUW). Les repères des autres rangs restent à construire, avec `lolcoach-collecte classement <RANG> <DIVISION>`.
- 194 parties de test : les pourcentages ont une marge d'environ ±3 points.
- Les parties couvrent un an de patchs. Avec plus de données, entraîner sur le patch en cours (`lolcoach-entrainer --patch`) sera plus juste.
- La justesse des conseils eux-mêmes (est-ce le bon conseil pour ce joueur ?) n'a pas été évaluée par un coach humain.

## Refaire cette évaluation

```bash
lolcoach-collecte verifier "Pseudo#EUW"
lolcoach-collecte classement GOLD II
lolcoach-collecte parties --joueurs 60 --par-joueur 15
lolcoach-entrainer                     # affiche les mesures des modèles
python benchmarks/stabilite.py         # stabilité des statistiques du diagnostic
```
