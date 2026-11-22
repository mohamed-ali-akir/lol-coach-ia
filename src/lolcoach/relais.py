"""Relais entre une session Claude du cloud et ton PC, pour créer des fiches avec `agy`.

Une session du cloud n'a ni `agy` ni ton compte Google. Elle confie donc le travail à ton PC
par l'intermédiaire de GitHub :

    cloud : lolcoach-relais demander "https://www.youtube.com/watch?v=…" --role JUNGLE
            → pousse une branche orpheline `relais-fiches-<vidéo>-<rôle>` (commit « demande »)
    PC    : lolcoach-relais surveiller        (toutes les 20 s, ou outils\\relais_lolcoach.cmd)
            → lance l'IA des fiches (LOLCOACH_FICHES=agy), ajoute un commit « réponse »
    cloud : lolcoach-relais lire
            → écrit les fiches « à valider » dans connaissances/, puis marque la branche
              comme lue (commit « lu.json ») ; le PC la supprime à son passage suivant
              (une session du cloud a le droit de pousser des branches, pas d'en supprimer)

Choix de sécurité :
- le PC n'exécute rien qui vienne de la demande : il ne lit que l'identifiant de la vidéo
  (11 caractères) et le rôle, reconstruit lui-même le lien YouTube, et lance toujours la même
  commande (celle de LOLCOACH_FICHES sur le PC) ;
- tout passe par la plomberie de git (objets et références) : le relais ne touche ni à ta
  branche courante ni à tes fichiers ;
- une demande n'est traitée qu'une fois (état dans .git/lolcoach-relais/), une seule à la
  fois, la plus ancienne d'abord ; un relais coupé en plein travail répond « erreur » ;
- les commits du relais portent l'identité « lolcoach-relais », sans adresse personnelle.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from lolcoach.connaissances import ROLES, KnowledgeError, load_cards, write_card
from lolcoach.fiches_cli import ask_for_cards, canonical_url, cards_from_answer, youtube_id
from lolcoach.ia.fournisseurs import AIError

VERSION = 1
DEFAULT_PREFIX = "relais"
DEFAULT_REMOTE = "origin"
DEFAULT_INTERVAL_S = 20.0
MAX_ANSWER_BYTES = 200_000  # une réponse de 8 fiches fait quelques Ko
MAX_REQUEST_BYTES = 10_000
GIT_TIMEOUT_S = 120.0
PENDING = 3  # code de sortie de `lire` quand des demandes attendent encore
IDENTITY = {
    "GIT_AUTHOR_NAME": "lolcoach-relais", "GIT_AUTHOR_EMAIL": "relais@lolcoach.invalid",
    "GIT_COMMITTER_NAME": "lolcoach-relais", "GIT_COMMITTER_EMAIL": "relais@lolcoach.invalid",
}
_VIDEO_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

Asker = Callable[[str, str], tuple[dict[str, Any], str]]
Clock = Callable[[], datetime]


class RelayError(Exception):
    """Problème avec git, GitHub ou une demande."""


def _now() -> datetime:
    return datetime.now(UTC)


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def prefix() -> str:
    value = os.environ.get("LOLCOACH_RELAIS_PREFIXE", "").strip() or DEFAULT_PREFIX
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_/-]{0,40}", value) or ".." in value:
        raise RelayError(f"LOLCOACH_RELAIS_PREFIXE invalide : {value!r}")
    return value


def branch_for(video: str, role: str) -> str:
    """Une branche par vidéo et par rôle : redemander la même vidéo remplace la demande."""
    if not _VIDEO_RE.match(video) or role not in ROLES:
        raise RelayError("vidéo ou rôle invalide")
    return f"{prefix()}-fiches-{video}-{role.lower()}"


# --- git, sans toucher à l'arbre de travail ---

class Git:
    def __init__(self, cwd: Path | None = None, remote: str | None = None) -> None:
        self.cwd = cwd or Path.cwd()
        self.remote = remote or os.environ.get("LOLCOACH_RELAIS_DISTANT", "").strip() \
            or DEFAULT_REMOTE

    def run(self, *args: str, data: bytes | None = None) -> str:
        try:
            done = subprocess.run(
                ["git", *args], cwd=self.cwd, input=data, capture_output=True,
                timeout=GIT_TIMEOUT_S, check=False, env={**os.environ, **IDENTITY})
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RelayError(f"git {args[0]} : {exc}") from exc
        if done.returncode != 0:
            detail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            raise RelayError(f"git {args[0]} a échoué : {detail[0][:300]}")
        return done.stdout.decode("utf-8", "replace")

    def git_dir(self) -> Path:
        return (self.cwd / self.run("rev-parse", "--git-dir").strip()).resolve()

    def heads(self) -> dict[str, str]:
        """Branches de demande sur le dépôt distant : {branche: sha}."""
        out = self.run("ls-remote", "--heads", self.remote, f"refs/heads/{prefix()}-fiches-*")
        found = {}
        for line in out.splitlines():
            sha, _, ref = line.partition("\t")
            name = ref.removeprefix("refs/heads/")
            if re.fullmatch(r"[0-9a-f]{40}", sha) and ref.startswith("refs/heads/"):
                found[name] = sha
        return found

    def fetch(self, branch: str) -> str:
        """Récupère la branche dans un espace de références à part : aucune branche locale
        n'est créée ni modifiée."""
        local = f"refs/lolcoach-relais/{branch}"
        self.run("fetch", "--no-tags", "--quiet", "--depth=3", self.remote,
                 f"+refs/heads/{branch}:{local}")
        return self.run("rev-parse", local).strip()

    def read(self, sha: str, name: str, limit: int) -> bytes | None:
        try:
            size = int(self.run("cat-file", "-s", f"{sha}:{name}").strip())
        except (RelayError, ValueError):
            return None
        if size > limit:
            raise RelayError(f"{name} trop volumineux ({size} octets)")
        done = subprocess.run(["git", "cat-file", "blob", f"{sha}:{name}"], cwd=self.cwd,
                              capture_output=True, timeout=GIT_TIMEOUT_S, check=False)
        return done.stdout if done.returncode == 0 else None

    def commit(self, files: dict[str, bytes], message: str, parent: str | None) -> str:
        entries = []
        for name, content in sorted(files.items()):
            blob = self.run("hash-object", "-w", "--stdin", data=content).strip()
            entries.append(f"100644 blob {blob}\t{name}\n")
        tree = self.run("mktree", data="".join(entries).encode()).strip()
        args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
        return self.run(*args).strip()

    def push(self, sha: str, branch: str, force: bool = False) -> None:
        self.run("push", "--quiet", *(["--force"] if force else []), self.remote,
                 f"{sha}:refs/heads/{branch}")

    def delete(self, branch: str) -> None:
        self.run("push", "--quiet", self.remote, "--delete", f"refs/heads/{branch}")
        with contextlib.suppress(RelayError):
            self.run("update-ref", "-d", f"refs/lolcoach-relais/{branch}")


def _json(raw: bytes | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


@dataclass
class Request:
    branch: str
    sha: str
    request_id: str
    video: str
    role: str
    date: str


def parse_request(branch: str, sha: str, data: dict[str, Any] | None) -> Request:
    """Ne garde de la demande que ce qui est vérifiable ; tout le reste est ignoré."""
    if data is None:
        raise RelayError("demande.json absent ou illisible")
    request_id, video, role = data.get("id_demande"), data.get("video"), data.get("role")
    if data.get("type") != "fiches":
        raise RelayError(f"type de demande inconnu : {data.get('type')!r}")
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.match(request_id):
        raise RelayError("identifiant de demande invalide")
    if not isinstance(video, str) or not isinstance(role, str):
        raise RelayError("vidéo ou rôle manquant")
    if branch != branch_for(video, role):
        raise RelayError("la branche ne correspond pas à la demande")
    date = data.get("date")
    return Request(branch, sha, request_id, video, role, date if isinstance(date, str) else "")


# --- côté cloud ---

def request_cards(git: Git, url: str, role: str, clock: Clock = _now) -> tuple[str, str]:
    """Pousse une demande. Renvoie (branche, identifiant de la demande)."""
    role = role.upper()
    video = youtube_id(url)
    branch = branch_for(video, role)
    moment = clock()
    request_id = f"{moment.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"
    request = {"version": VERSION, "type": "fiches", "id_demande": request_id,
               "video": video, "role": role, "date": moment.isoformat()}
    sha = git.commit({"demande.json": json.dumps(request, indent=1).encode()},
                     f"Demande fiches {request_id} : {video} ({role})", None)
    git.push(sha, branch, force=True)  # remplace une demande plus ancienne pour la même vidéo
    return branch, request_id


def _mark_read(git: Git, request: Request, files: dict[str, bytes]) -> None:
    marker = {"id_demande": request.request_id, "lu": _now().isoformat()}
    sha = git.commit({**files, "lu.json": json.dumps(marker).encode()},
                     f"Lu {request.request_id}", request.sha)
    git.push(sha, request.branch)


def _is_read(git: Git, request: Request) -> bool:
    marker = _json(git.read(request.sha, "lu.json", MAX_REQUEST_BYTES))
    return marker is not None and marker.get("id_demande") == request.request_id


@dataclass
class Outcome:
    branch: str
    status: str  # "lue", "erreur", "en_attente", "illisible"
    detail: str = ""
    cards: int = 0


def read_answers(git: Git, keep: bool = False, folder: Path | None = None) -> list[Outcome]:
    """Écrit les fiches des réponses arrivées, puis supprime leurs branches ou, si le dépôt
    le refuse (session du cloud), les marque comme lues. Relancé, il ne crée jamais deux fois
    la même fiche."""
    outcomes = []
    cards_now = load_cards(folder)[0]
    existing = {card.id for card in cards_now}
    already = {(card.source, card.title) for card in cards_now}
    for branch in sorted(git.heads()):
        try:
            sha = git.fetch(branch)
            raw_request = git.read(sha, "demande.json", MAX_REQUEST_BYTES) or b""
            request = parse_request(branch, sha, _json(raw_request))
            raw_answer = git.read(sha, "reponse.json", MAX_ANSWER_BYTES) or b""
            answer = _json(raw_answer)
            if _is_read(git, request):
                continue  # déjà lue ; le PC supprimera la branche
        except RelayError as exc:
            outcomes.append(Outcome(branch, "illisible", str(exc)))
            continue
        if answer is None or answer.get("id_demande") != request.request_id:
            outcomes.append(Outcome(branch, "en_attente"))
            continue
        if answer.get("statut") != "ok" or not isinstance(answer.get("reponse"), dict):
            outcomes.append(Outcome(branch, "erreur", str(answer.get("erreur", ""))[:300]))
        else:
            try:
                cards = cards_from_answer(answer["reponse"], f"youtube:{request.video}",
                                          request.role, existing)
            except KnowledgeError as exc:
                outcomes.append(Outcome(branch, "erreur", f"réponse inutilisable : {exc}"))
            else:
                fresh = [c for c in cards if (c.source, c.title) not in already]
                for card in fresh:
                    write_card(card, folder)
                    existing.add(card.id)
                    already.add((card.source, card.title))
                outcomes.append(Outcome(branch, "lue", ", ".join(c.id for c in fresh),
                                        len(fresh)))
        if not keep:
            try:
                git.delete(branch)
            except RelayError:
                _mark_read(git, request, {"demande.json": raw_request,
                                          "reponse.json": raw_answer})
    return outcomes


# --- côté PC ---

class State:
    """Demandes déjà vues ou traitées, dans .git/lolcoach-relais/etat.json."""

    def __init__(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "etat.json"
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        self.requests: dict[str, dict[str, Any]] = data.get("demandes", {}) \
            if isinstance(data, dict) else {}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"demandes": self.requests}, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def mark(self, request_id: str, **values: Any) -> None:
        self.requests[request_id] = {**values, "maj": _now().isoformat()}
        self.save()


def _answer(request: Request, status: str, seconds: float, **extra: Any) -> bytes:
    body = {"version": VERSION, "id_demande": request.request_id, "type": "fiches",
            "statut": status, "secondes": round(seconds), "date": _now().isoformat(), **extra}
    return json.dumps(body, ensure_ascii=False, indent=1).encode()


def _respond(git: Git, request: Request, payload: bytes, request_raw: bytes) -> None:
    sha = git.commit({"demande.json": request_raw, "reponse.json": payload},
                     f"Réponse fiches {request.request_id}", request.sha)
    git.push(sha, request.branch)  # avance rapide : refusée si la demande a été remplacée


def pending(git: Git, state: State) -> list[tuple[Request, bytes]]:
    """Demandes à traiter, de la plus ancienne à la plus récente."""
    found = []
    for branch, remote_sha in git.heads().items():
        if state.requests.get(f"branche:{branch}", {}).get("sha") == remote_sha:
            continue  # déjà examinée telle quelle
        try:
            sha = git.fetch(branch)
            raw = git.read(sha, "demande.json", MAX_REQUEST_BYTES) or b""
            request = parse_request(branch, sha, _json(raw))
            answered = _json(git.read(sha, "reponse.json", MAX_ANSWER_BYTES))
        except RelayError as exc:
            _log(f"{branch} ignorée : {exc}")
            state.mark(f"branche:{branch}", sha=remote_sha)
            continue
        try:
            if _is_read(git, request):  # lue par le cloud, qui n'a pas pu la supprimer
                git.delete(branch)
                _log(f"{branch} lue par le cloud : supprimée.")
                continue
        except RelayError as exc:
            _log(f"{branch} : suppression impossible ({exc})")
        if (answered and answered.get("id_demande") == request.request_id) \
                or state.requests.get(request.request_id, {}).get("etat") == "fait":
            state.mark(f"branche:{branch}", sha=remote_sha)
            continue
        found.append((request, raw))
    return sorted(found, key=lambda item: (item[0].date, item[0].request_id))


def process_one(git: Git, state: State, ask: Asker = ask_for_cards) -> str | None:
    """Traite la plus ancienne demande en attente. Renvoie son identifiant, ou None."""
    waiting = pending(git, state)
    if not waiting:
        return None
    request, raw = waiting[0]
    state.mark(request.request_id, etat="en_cours", branche=request.branch)
    _log(f"Demande {request.request_id} : fiches {request.role} pour la vidéo {request.video}")
    start = time.monotonic()
    try:
        answer, provider = ask(canonical_url(request.video), request.role)
        payload = _answer(request, "ok", time.monotonic() - start, fournisseur=provider,
                          reponse=answer)
        if len(payload) > MAX_ANSWER_BYTES:
            raise AIError(f"réponse trop volumineuse ({len(payload)} octets)")
        count = len(answer.get("fiches") or []) if isinstance(answer, dict) else 0
        summary = f"{count} fiche(s) proposée(s) par {provider}"
    except (AIError, KnowledgeError, ValueError, OSError) as exc:
        payload = _answer(request, "erreur", time.monotonic() - start, erreur=str(exc)[:500])
        summary = f"erreur : {exc}"
    try:
        _respond(git, request, payload, raw)
    except RelayError as exc:
        _log(f"Réponse non envoyée ({exc}) : la demande a peut-être été remplacée.")
        state.mark(request.request_id, etat="fait", branche=request.branch, envoyee=False)
        return request.request_id
    state.mark(request.request_id, etat="fait", branche=request.branch)
    _log(f"Réponse envoyée en {time.monotonic() - start:.0f} s : {summary}")
    return request.request_id


def recover(git: Git, state: State) -> None:
    """Une demande coupée en plein travail (PC éteint, Ctrl+C) reçoit une réponse « erreur » :
    le cloud la redemandera."""
    for request_id, info in list(state.requests.items()):
        if info.get("etat") != "en_cours" or not isinstance(info.get("branche"), str):
            continue
        try:
            sha = git.fetch(info["branche"])
            raw = git.read(sha, "demande.json", MAX_REQUEST_BYTES) or b""
            request = parse_request(info["branche"], sha, _json(raw))
            if request.request_id == request_id:
                _respond(git, request, _answer(request, "erreur", 0, erreur=(
                    "Relais interrompu pendant le travail. Refais la demande.")), raw)
        except RelayError as exc:
            _log(f"Demande interrompue {request_id} : {exc}")
        state.mark(request_id, etat="fait", branche=info["branche"])


@contextlib.contextmanager
def single_instance(folder: Path) -> Iterator[None]:
    lock = folder / "relais.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RelayError(f"Un relais tourne déjà sur ce dépôt (sinon, supprime {lock}).") from exc
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _keep_awake(on: bool) -> None:
    """Empêche la mise en veille de Windows tant que le relais tourne (l'écran s'éteint)."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | (es_system_required if on else 0))


def watch(git: Git, interval: float, once: bool, ask: Asker = ask_for_cards,
          stay_awake: bool = True) -> None:
    folder = git.git_dir() / "lolcoach-relais"
    folder.mkdir(parents=True, exist_ok=True)
    state = State(folder)
    with single_instance(folder):
        _log(f"Relais : surveille {git.remote}/{prefix()}-fiches-* toutes les "
             f"{interval:.0f} s. Ctrl+C pour arrêter.")
        if stay_awake and not once:
            _keep_awake(True)
        try:
            recover(git, state)
            while True:
                try:
                    while process_one(git, state, ask):
                        pass
                except RelayError as exc:
                    _log(f"Problème passager : {exc}")
                if once:
                    return
                time.sleep(interval)
        except KeyboardInterrupt:
            _log("Relais arrêté.")
        finally:
            if stay_awake and not once:
                _keep_awake(False)


# --- ligne de commande ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lolcoach-relais", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="commande", required=True)
    p_ask = sub.add_parser("demander", help="(cloud) confier une vidéo au PC")
    p_ask.add_argument("url")
    p_ask.add_argument("--role", type=str.upper, choices=ROLES, default="JUNGLE")
    p_read = sub.add_parser("lire", help="(cloud) récupérer les fiches arrivées")
    p_read.add_argument("--garder", action="store_true", help="ne pas supprimer les branches")
    p_watch = sub.add_parser("surveiller", help="(PC) répondre aux demandes")
    p_watch.add_argument("--intervalle", type=float, default=DEFAULT_INTERVAL_S)
    p_watch.add_argument("--une-fois", action="store_true", help="un seul passage")
    p_watch.add_argument("--laisser-dormir", action="store_true")
    sub.add_parser("etat", help="demandes en cours sur le dépôt distant")
    return parser


def _run(args: argparse.Namespace) -> int:
    load_dotenv()
    git = Git()
    if args.commande == "demander":
        branch, request_id = request_cards(git, args.url, args.role)
        print(f"Demande {request_id} envoyée sur {git.remote}/{branch}.")
        print("Le relais du PC la prend dans les 20 s s'il tourne (2 à 5 min pour une vidéo). "
              "Ensuite : lolcoach-relais lire")
        return 0
    if args.commande == "lire":
        outcomes = read_answers(git, keep=args.garder)
        for o in outcomes:
            text = {"lue": (f"{o.cards} fiche(s) à valider : {o.detail}" if o.cards
                            else "rien de nouveau (fiches déjà importées)"),
                    "en_attente": "en attente du PC", "erreur": f"erreur du PC : {o.detail}",
                    "illisible": f"illisible : {o.detail}"}[o.status]
            print(f"{o.branch} : {text}")
        if not outcomes:
            print("Aucune demande en cours.")
        elif any(o.status == "lue" and o.cards for o in outcomes):
            print("Relis-les, puis : lolcoach-fiches valider <id>")
        return PENDING if any(o.status == "en_attente" for o in outcomes) else 0
    if args.commande == "etat":
        heads = git.heads()
        for branch in sorted(heads):
            print(branch)
        if not heads:
            print("Aucune demande en cours.")
        return 0
    watch(git, args.intervalle, args.une_fois, stay_awake=not args.laisser_dormir)
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = _run(args)
    except (RelayError, KnowledgeError, AIError, OSError) as exc:
        sys.exit(f"Erreur : {exc}")
    sys.exit(code)


if __name__ == "__main__":
    main()
