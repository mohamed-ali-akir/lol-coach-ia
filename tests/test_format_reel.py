"""Réponses au format réel documenté de l'API Riot : bien plus de champs et de types
d'événements que ce que le projet utilise. Rien de tout cela ne doit gêner l'analyse.

Ces tests ne remplacent pas un essai avec une vraie clé (`lolcoach-collecte verifier`) :
ils garantissent seulement que les champs et événements supplémentaires connus sont tolérés.
"""

import copy

from conftest import simulated
from test_pipeline import FakeRiot

from lolcoach import db
from lolcoach.collecte import verification
from lolcoach.collecte.parse import parse_match
from lolcoach.collecte.timeline import HANDLED_EVENT_TYPES, parse_timeline
from lolcoach.riot import RiotAuthError, RiotNotFound


def realistic(seed=0):
    """Une partie simulée, enrichie de tout ce qu'une vraie réponse contient en plus."""
    match, timeline = copy.deepcopy(simulated(seed=seed))
    info = match["info"]
    info.update({
        "gameMode": "CLASSIC", "gameType": "MATCHED_GAME", "mapId": 11, "platformId": "EUW1",
        "gameStartTimestamp": info["gameCreation"] + 90_000, "gameId": 7_000_000_001,
        "teams": [{"teamId": team, "win": team == 100, "bans": [{"championId": 1, "pickTurn": 1}],
                   "objectives": {"baron": {"first": False, "kills": 0}}} for team in (100, 200)],
    })
    for player in info["participants"]:
        player.update({
            "riotIdGameName": "Un Pseudo", "riotIdTagline": "EUW", "summonerLevel": 312,
            "individualPosition": player["teamPosition"], "lane": "MIDDLE", "role": "SOLO",
            "item0": 3157, "item6": 3364, "summoner1Id": 4, "summoner2Id": 14,
            "perks": {"statPerks": {"defense": 5002}, "styles": [{"style": 8100}]},
            "allInPings": 3, "firstBloodKill": False, "champLevel": 16, "timePlayed": 1800,
        })
        player["challenges"].update({
            "kda": 2.75, "killParticipation": 0.61, "visionScorePerMinute": 0.83,
            "teamDamagePercentage": 0.24, "laneMinionsFirst10Minutes": 71,
            "controlWardsPlaced": 2, "legendaryItemUsed": [3157, 4645],
        })
    frames = timeline["info"]["frames"]
    for frame in frames:
        for pf in frame["participantFrames"].values():
            pf.update({
                "goldPerSecond": 0, "timeEnemySpentControlled": 1234,
                "championStats": {"abilityPower": 120, "armor": 44, "health": 1800},
                "damageStats": {"magicDamageDone": 9000, "totalDamageTaken": 12000},
            })
    extra = [
        {"type": "PAUSE_END", "timestamp": 0, "realTimestamp": info["gameCreation"]},
        {"type": "ITEM_PURCHASED", "timestamp": 4_000, "participantId": 3, "itemId": 1056},
        {"type": "SKILL_LEVEL_UP", "timestamp": 5_000, "participantId": 3, "skillSlot": 1,
         "levelUpType": "NORMAL"},
        {"type": "LEVEL_UP", "timestamp": 95_000, "participantId": 3, "level": 2},
        {"type": "ITEM_UNDO", "timestamp": 96_000, "participantId": 3, "beforeId": 1056,
         "afterId": 0, "goldGain": 400},
        {"type": "CHAMPION_SPECIAL_KILL", "timestamp": 200_000, "killerId": 8,
         "killType": "KILL_FIRST_BLOOD", "position": {"x": 7000, "y": 7000}},
        {"type": "OBJECTIVE_BOUNTY_PRESTART", "timestamp": 900_000, "teamId": 200,
         "actualStartTime": 915_000},
        {"type": "FEAT_UPDATE", "timestamp": 300_000, "teamId": 100, "featType": 1,
         "featValue": 1},
        {"type": "UN_TYPE_APPARU_EN_2031", "timestamp": 400_000, "donnee": {"x": 1}},
    ]
    documented = [
        # Mort avec le détail des dégâts, sans liste d'assistances (champ facultatif).
        {"type": "CHAMPION_KILL", "timestamp": 210_000, "killerId": 8, "victimId": 3,
         "position": {"x": 7100, "y": 7200}, "bounty": 300, "shutdownBounty": 0,
         "killStreakLength": 1,
         "victimDamageReceived": [{"basic": True, "magicDamage": 0, "name": "Ahri",
                                   "participantId": 8, "physicalDamage": 90,
                                   "spellName": "ahribasicattack", "spellSlot": 64,
                                   "trueDamage": 0, "type": "OTHER"}],
         "victimDamageDealt": [], "victimTeamfightDamageDealt": []},
        # Exécution : aucun champion n'a porté le coup fatal.
        {"type": "CHAMPION_KILL", "timestamp": 220_000, "killerId": 0, "victimId": 9,
         "position": {"x": 1000, "y": 1000}, "bounty": 0},
        # Inhibiteur : pas de `towerType`.
        {"type": "BUILDING_KILL", "timestamp": 1_500_000, "killerId": 1, "teamId": 200,
         "buildingType": "INHIBITOR_BUILDING", "laneType": "MID_LANE",
         "assistingParticipantIds": [2, 3], "position": {"x": 11000, "y": 11000}, "bounty": 0},
        # Dragon ancestral, tué sans champion crédité.
        {"type": "ELITE_MONSTER_KILL", "timestamp": 1_600_000, "killerId": 0,
         "killerTeamId": 100, "monsterType": "DRAGON", "monsterSubType": "ELDER_DRAGON",
         "position": {"x": 9800, "y": 4400}, "bounty": 0},
        {"type": "DRAGON_SOUL_GIVEN", "timestamp": 1_400_000, "name": "Mountain", "teamId": 100},
        # Balise parasite des timelines récentes.
        {"type": "WARD_PLACED", "timestamp": 230_000, "creatorId": 0, "wardType": "UNDEFINED"},
    ]
    frames[1]["events"] = extra + documented + frames[1]["events"]
    frames[-1]["events"].append({"type": "GAME_END", "timestamp": frames[-1]["timestamp"],
                                 "winningTeam": 100, "gameId": 7_000_000_001,
                                 "realTimestamp": info["gameCreation"] + 1_800_000})
    return match, timeline


def test_real_format_extras_do_not_disturb_parsing():
    plain_match, plain_timeline = simulated()
    match, timeline = realistic()

    assert parse_match(match) == parse_match(plain_match)  # champs en plus : même résultat
    _, rows = parse_match(match)
    team_of = {r["participant_id"]: r["team_id"] for r in rows}
    puuid_of = {r["participant_id"]: r["puuid"] for r in rows}
    frames, events = parse_timeline(timeline, "SIM1_1", team_of, puuid_of)
    plain_frames, plain_events = parse_timeline(plain_timeline, "SIM1_1", team_of, puuid_of)

    assert frames == plain_frames
    assert len(events) == len(plain_events) + 5  # les 5 événements documentés et utilisables
    by_time = {e["timestamp_ms"]: e for e in events}
    assert by_time[210_000]["actor_id"] == 8 and by_time[210_000]["assists_mask"] == 0
    assert by_time[220_000]["actor_id"] == 0 and by_time[220_000]["team_id"] == 100
    inhibitor = by_time[1_500_000]
    assert (inhibitor["type"], inhibitor["kind"], inhibitor["sub_kind"]) == (
        db.EVENT_BUILDING, "INHIBITOR_BUILDING", None)
    assert inhibitor["assists_mask"] == db.assists_mask([2, 3])
    elder = by_time[1_600_000]
    assert (elder["team_id"], elder["actor_id"], elder["sub_kind"]) == (100, 0, "ELDER_DRAGON")
    assert by_time[1_400_000]["type"] == db.EVENT_SOUL


def live_riot(match, timeline, **overrides):
    riot = FakeRiot({"p1": ["SIM1_1"]}, {"SIM1_1": match}, {"SIM1_1": timeline})
    for name, value in overrides.items():
        setattr(riot, name, value)
    return riot


def test_live_check_reports_every_step_and_unknown_event_types():
    match, timeline = realistic()
    check = verification.check_live(live_riot(match, timeline), "Moi#EUW")

    assert check.ok
    steps = {step: detail for step, _, detail in check.steps}
    assert list(steps) == ["Compte trouvé", "Rang lu", "Liste des parties classées",
                           "Partie analysée", "Timeline trouvée", "Timeline analysée",
                           "Photos", "Événements gardés"]
    assert steps["Rang lu"] == "PLATINUM III" and steps["Partie analysée"] == "SIM1_1"
    assert "espacées de" in steps["Photos"] and "KILL" in steps["Événements gardés"]
    unknown = next(w for w in check.warnings if "ignorés" in w)
    assert "ITEM_PURCHASED × 1" in unknown and "UN_TYPE_APPARU_EN_2031 × 1" in unknown
    assert not any(name in unknown for name in HANDLED_EVENT_TYPES)
    text = verification.render(check)
    assert "OK    Timeline analysée" in text and text.endswith("lancer la collecte.")


def test_live_check_explains_each_failure():
    match, timeline = realistic()

    def expired(*_args, **_kwargs):
        raise RiotAuthError("Clé Riot refusée : elle est absente, invalide ou expirée.")

    def unknown_account(*_args, **_kwargs):
        raise RiotNotFound("x")

    for riot, expected in (
        (live_riot(match, timeline, account_by_riot_id=expired), "Clé Riot refusée"),
        (live_riot(match, timeline, account_by_riot_id=unknown_account), "Appel à l'API Riot"),
        (live_riot(match, timeline, matches_by_player={}), "aucune partie classée récente"),
        (live_riot(match, {"info": {"frames": "?"}}), "format de réponse inattendu"),
        (live_riot(match, timeline, timelines={}), "ne fournit pas la timeline"),
        (live_riot({"info": {}}, timeline), "aucune des parties récentes n'est exploitable"),
    ):
        check = verification.check_live(riot, "Moi#EUW")
        text = verification.render(check)
        assert not check.ok and expected in text and "Un point bloque" in text

    one_frame = copy.deepcopy(timeline)
    del one_frame["info"]["frames"][1:]
    short = verification.check_live(live_riot(match, one_frame), "Moi#EUW")
    assert not short.ok and "trop peu" in verification.render(short)

    bad_id = verification.check_live(live_riot(match, timeline), "SansTag")
    assert not bad_id.ok and "Riot ID invalide" in verification.render(bad_id)


def test_raw_event_types_survive_malformed_timelines():
    for broken in ({}, {"info": None}, {"info": {"frames": "?"}},
                   {"info": {"frames": [None, {"events": "?"}, {"events": [None, 3]}]}}):
        assert not verification._raw_event_types(broken)
    counted = verification._raw_event_types(
        {"info": {"frames": [{"events": [{"type": "LEVEL_UP"}, {"sans": "type"}]}]}})
    assert counted == {"LEVEL_UP": 1, "None": 1}


def test_verifier_command_prints_the_check_and_fails_when_blocked(monkeypatch, capsys):
    import pytest

    from lolcoach.collecte import cli

    match, timeline = realistic()
    monkeypatch.setenv("RIOT_API_KEY", "RGAPI-essai")

    class Client:
        def __init__(self, *_args, **_kwargs):
            self.riot = live_riot(match, timeline, **self.overrides)

        def __enter__(self):
            return self.riot

        def __exit__(self, *_exc):
            return None

        overrides: dict = {}

    monkeypatch.setattr(cli, "RiotClient", Client)
    monkeypatch.setattr(cli, "init_db", lambda *_a, **_k: pytest.fail("rien ne doit être écrit"))
    monkeypatch.setattr("sys.argv", ["lolcoach-collecte", "verifier", "Moi#EUW"])
    cli.main()
    assert "Tout fonctionne" in capsys.readouterr().out

    Client.overrides = {"matches_by_player": {}}
    with pytest.raises(SystemExit) as stop:
        cli.main()
    assert stop.value.code == 1 and "Un point bloque" in capsys.readouterr().out


def test_live_check_flags_missing_optional_stats():
    match, timeline = realistic()
    for player in match["info"]["participants"]:
        del player["wardsPlaced"]
        player["challenges"] = None
    check = verification.check_live(live_riot(match, timeline), "Moi#EUW")
    assert check.ok  # facultatif : signalé, pas bloquant
    warning = next(w for w in check.warnings if "facultatives" in w)
    assert "wards_placed" in warning and "solo_kills" in warning
