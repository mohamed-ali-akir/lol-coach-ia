# Qualité sur de vraies parties

Ce document dit ce que valent les modèles et le diagnostic **sur de vraies parties**, et ce que ces données ont permis de corriger. Les chiffres de la démo viennent de parties simulées et ne disent rien du jeu réel ; ceux-ci, si.

## Les données (22/11/2026)

| | |
|---|---|
| Parties | 1 659 parties classées solo/duo (file 420), serveur EUW, toutes avec leur timeline |
| Rangs | Or II (986 parties), Argent II (338), Platine II (335), joueurs tirés du classement officiel |
| Joueurs suivis | 128, 12 à 15 dernières parties chacun |
| Période | patchs 15.5 à 16.23 : certains joueurs jouent peu, leurs dernières parties remontent à un an |
| Clé | clé de développement Riot, environ 70 minutes de collecte au total, aucun refus de Riot après correction du limiteur |

Les chiffres ci-dessous sont ceux des 1 659 parties ; la première évaluation, sur les 966 parties Or seules, est indiquée quand elle diffère.

Aucune donnée de joueur n'est versionnée : la base et l'archive restent sur la machine qui a collecté.

## La chaîne de collecte

`lolcoach-collecte verifier` passe toutes les étapes sur une vraie partie : compte, rang, liste des parties, partie, timeline, photos (écart de 37 à 60 s, puis de moins d'une seconde pour la dernière), événements. Riot envoie en plus des types d'événements que le projet ignore volontairement : achats, ventes et destructions d'objets, niveaux, sorts, pauses, fin de partie, kills spéciaux.

Sur la collecte complète : statistiques facultatives toutes présentes, aucune partie écartée pour un format inattendu, 4 parties écartées comme remakes ou sans rôles complets.

## Probabilité de victoire

Entraînée sur les 1 327 parties les plus anciennes, testée sur les 332 plus récentes.

| | Modèle | Référence naïve |
|---|---|---|
| Bonnes prédictions | 73,4 % (Or seul : 72,5 %) | 48,9 % |
| Score de Brier (plus bas = mieux) | 0,171 (Or seul : 0,177) | 0,250 |
| Écart de calibration | 1,4 % (Or seul : 2,8 %) | |

| Moment | 0-5 | 5-10 | 10-15 | 15-20 | 20-25 | 25-30 | 30-35 | 35+ min |
|---|---|---|---|---|---|---|---|---|
| Bonnes prédictions | 56,5 % | 68,0 % | 75,7 % | 78,8 % | 79,2 % | 78,7 % | 80,1 % | 79,7 % |

Le modèle est utile et ses probabilités sont fiables : quand il annonce 70 %, l'équipe gagne à peu près 7 fois sur 10. Il progresse avec les données : sur les seules parties Or, il tombait à 67,8 % après 35 minutes.

## Draft

51,8 % de bonnes prédictions contre 49,4 % pour la référence (Or seul : 55,2 % contre 52,6 %), mais une perte logarithmique moins bonne que la référence. Le modèle n'est donc **pas enregistré**, et son classement des champions n'est pas affiché. Avec 446 parties, il faisait même moins bien que la référence en précision (54,5 % contre 58,0 %). Un modèle de draft demande des dizaines de milliers de parties. Le coach ne s'en sert pas.

Règle ajoutée après cet essai : un modèle qui ne prédit pas mieux que la référence naïve n'est ni enregistré par `lolcoach-entrainer`, ni utilisé par le coach (`useful` dans `modele/winprob.py` et `modele/draft.py`).

## Repères

16 590 performances individuelles ; chacun des trois rangs a ses propres repères. Ce qui distingue le plus les victoires des défaites, à tous les postes : les morts par 10 minutes et le temps passé mort, puis l'écart d'or ou d'XP à 15 minutes face à l'adversaire direct (le farm par minute en jungle). Farm par minute médian, tous rangs : 6,77 en top, 6,50 en jungle, 6,70 en mid, 7,05 en ADC.

## Le diagnostic mesure-t-il quelque chose ?

Test (`analyse/stability.py`, détail avec `benchmarks/stabilite.py`) : les parties de chaque joueur suivi sont coupées en deux moitiés, une partie sur deux. Si une statistique mesure un vrai trait du joueur, les deux moitiés classent les joueurs dans le même ordre (corrélation de Spearman).

| Statistique | Or seul (48 joueurs) | Trois rangs (72 joueurs) |
|---|---|---|
| Balises de contrôle achetées | 0,91 | 0,89 |
| Score de vision par minute | 0,78 | 0,79 |
| Part des dégâts de l'équipe | 0,77 | 0,67 |
| Farm par minute | 0,67 | 0,65 |
| Sbires à 10 minutes | 0,47 | 0,55 |
| Morts par 10 minutes | 0,41 | 0,45 |
| Temps passé mort | 0,40 | 0,45 |
| Écart d'or à 15 min | 0,43 | 0,41 |
| Morts avant 15 minutes | 0,37 | 0,41 |
| Présence aux objectifs | 0,38 | 0,40 |
| Écart d'XP à 15 min | 0,32 | 0,32 |
| Participation aux kills | 0,33 | 0,28 |
| **Morts seul face à plusieurs adversaires** | **0,11** | 0,28 |
| *Seuil du hasard* | *0,24* | *0,20* |

Le diagnostic mesure des traits réels : toutes les statistiques dépassent le seuil du hasard, sauf une sur les seules parties Or. Celle-là, « morts seul face à plusieurs adversaires », est la plus fragile : sous le seuil avec 48 joueurs, juste au-dessus avec 72.

D'où le choix : **la stabilité est mesurée à chaque entraînement** et enregistrée avec les repères. Une statistique sous le seuil ne devient jamais une priorité du diagnostic ; elle reste affichée dans le rapport de chaque partie, où elle décrit un fait. Avec moins de 20 joueurs suivis, rien n'est jugé ni exclu. `lolcoach-entrainer` affiche le résultat. Une première version excluait cette statistique une fois pour toutes ; les données suivantes ont montré qu'une décision écrite en dur aurait été fausse.

Sur les 128 joueurs suivis ayant assez de parties à leur poste, 25 n'ont aucune priorité, 51 en ont une, 27 deux, 25 trois. Sur des joueurs simulés tous moyens, 85 % n'en avaient aucune : de vrais joueurs du même rang ont, eux, de vraies forces et faiblesses, ce que confirme le test ci-dessus.

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
| Priorité possible sur une statistique instable | stabilité mesurée à chaque entraînement ; une statistique instable n'est jamais une priorité |
| La file de joueurs mélangeait les rangs | option `lolcoach-collecte parties --rang` |

## Limites de cette évaluation

- Trois rangs (Argent, Or, Platine), un seul serveur (EUW). Les autres rangs se replient sur « tous rangs confondus » tant qu'ils n'ont pas été collectés (`lolcoach-collecte classement <RANG> <DIVISION>`, puis `parties --rang <RANG>`).
- 332 parties de test : les pourcentages ont une marge d'environ ±2,5 points.
- Les parties couvrent un an de patchs. Avec plus de données, entraîner sur le patch en cours (`lolcoach-entrainer --patch`) sera plus juste.
- La justesse des conseils eux-mêmes (est-ce le bon conseil pour ce joueur ?) n'a pas été évaluée par un coach humain.

## Refaire cette évaluation

```bash
lolcoach-collecte verifier "Pseudo#EUW"
lolcoach-collecte classement GOLD II
lolcoach-collecte parties --rang GOLD --joueurs 60 --par-joueur 15
lolcoach-entrainer                     # mesures des modèles et stabilité des statistiques
python benchmarks/stabilite.py         # détail de la stabilité, statistique par statistique
```
