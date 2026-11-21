"""Archive des réponses brutes de l'API Riot (JSON compressé).

Pourquoi : Riot efface les anciennes parties, et le format de la base évoluera. Avec les
réponses d'origine sur le disque, on peut tout recalculer des années plus tard sans
rien retélécharger (`lolcoach-collecte reimporter`).

Organisation : <racine>/<patch>/<match_id>.<type>.json.gz, avec type = match ou timeline.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lolcoach.riot import validate_match_id

log = logging.getLogger(__name__)

PATCH_RE = re.compile(r"^\d{1,3}\.\d{1,3}$")
KINDS = ("match", "timeline")


class RawArchive:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, patch: str, match_id: str, kind: str) -> Path:
        # Les noms viennent d'une API externe : on les valide avant d'en faire un chemin.
        if kind not in KINDS:
            raise ValueError(f"type d'archive inconnu : {kind}")
        folder = patch if PATCH_RE.fullmatch(patch) else "inconnu"
        return self.root / folder / f"{validate_match_id(match_id)}.{kind}.json.gz"

    def save(self, patch: str, match_id: str, kind: str, payload: dict[str, Any]) -> Path:
        """Écrit le fichier de façon atomique : jamais de fichier à moitié écrit."""
        path = self._path(patch, match_id, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), mtime=0)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".gz")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def load(self, patch: str, match_id: str, kind: str) -> dict[str, Any] | None:
        path = self._path(patch, match_id, kind)
        if not path.exists():
            return None
        return _read(path)

    def iter_matches(self) -> Iterator[tuple[dict[str, Any], dict[str, Any] | None]]:
        """Parcourt l'archive : (partie, timeline ou None), dans un ordre stable."""
        if not self.root.exists():
            return
        for match_path in sorted(self.root.glob("*/*.match.json.gz")):
            timeline_path = match_path.with_name(
                match_path.name.replace(".match.json.gz", ".timeline.json.gz")
            )
            try:
                match = _read(match_path)
                timeline = _read(timeline_path) if timeline_path.exists() else None
            except (OSError, EOFError, ValueError) as exc:
                # Un fichier abîmé ne doit pas empêcher de relire tous les autres.
                log.warning("Archive illisible, ignorée : %s (%s)", match_path.name, exc)
                continue
            yield match, timeline


def _read(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} : contenu inattendu")
    return payload
