import json
import re
import subprocess

import pytest

from lolcoach.client_lol.replay import Moment
from lolcoach.ia import fournisseurs as f
from lolcoach.ia.consignes import VISION_SCHEMA, cards_prompt, report_prompt, vision_prompt


@pytest.mark.parametrize("text", [
    '{"moments": []}',
    'Voici :\n```json\n{"moments": []}\n```\nBonne journée',
    json.dumps({"type": "result", "result": 'Analyse :\n```json\n{"moments": []}\n```'}),
    json.dumps({"status": "SUCCESS", "structured_output": {"moments": []}}),
    'bla {"moments": []} bla',
])
def test_extract_json(text):
    assert f.extract_json(text) == {"moments": []}


def test_extract_json_failure_and_text_envelopes():
    with pytest.raises(f.AIError):
        f.extract_json("pas de JSON ici")
    assert f.extract_text(json.dumps({"result": " Rapport "})) == "Rapport"
    assert f.extract_text("texte brut") == "texte brut"
    with pytest.raises(f.AIError):
        f.extract_text(json.dumps({"is_error": True, "result": "quota"}))


class FakeRunner:
    def __init__(self, stdout="", code=0):
        self.calls = []
        self.stdout, self.code = stdout, code
        self.files = {}

    def __call__(self, args, timeout):
        self.calls.append(args)
        for arg in args:  # lit les fichiers temporaires tant qu'ils existent
            for path in re.findall(r"(/\S+?\.(?:txt|json))\b", arg):
                with open(path, encoding="utf-8") as handle:
                    self.files[path.rsplit("/", 1)[-1]] = handle.read()
        return subprocess.CompletedProcess(args, self.code, self.stdout, "boom\nerreur finale")


def test_command_provider_vision(tmp_path):
    images = [tmp_path / "a b.png", tmp_path / "c.png"]
    runner = FakeRunner(json.dumps({"result": '{"moments": [{"minute": "1:00"}]}'}))
    provider = f.CommandProvider(
        "agy -p {prompt} --json-schema {schema_file} {images}", name="agy", runner=runner)
    answer = provider.look("Regarde ces images", images, VISION_SCHEMA)
    assert answer == {"moments": [{"minute": "1:00"}]}
    args = runner.calls[0]
    assert args[:2] == ["agy", "-p"] and "Regarde ces images" in args[2]
    assert str(images[0].resolve()) in args  # un chemin avec espace reste un seul argument
    assert json.loads(runner.files["schema.json"]) == VISION_SCHEMA


def test_long_prompt_goes_through_a_file():
    runner = FakeRunner("Rapport rédigé")
    provider = f.CommandProvider("claude -p {prompt}", runner=runner)
    assert provider.write("x" * (f.LONG_PROMPT + 1)) == "Rapport rédigé"
    assert "fichier" in runner.calls[0][2] and len(runner.calls[0][2]) < 500
    assert runner.files["consigne.txt"] == "x" * (f.LONG_PROMPT + 1)


def test_command_failure():
    provider = f.CommandProvider("agy -p {prompt}", runner=FakeRunner(code=3))
    with pytest.raises(f.AIError, match="code 3.*erreur finale"):
        provider.write("bonjour")
    with pytest.raises(f.AIError, match="vide"):
        f.CommandProvider("agy -p {prompt}", runner=FakeRunner("")).write("bonjour")
    with pytest.raises(f.AIError, match="introuvable"):
        f.CommandProvider("commande-qui-n-existe-pas-xyz {prompt}")


NPM_SHIM = ('@ECHO off\r\nGOTO start\r\n:find_dp0\r\nSET dp0=%~dp0\r\nEXIT /b\r\n:start\r\n'
            'SETLOCAL\r\nCALL :find_dp0\r\n'
            '"%dp0%\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe"   %*\r\n')


def test_resolve_command_windows_npm_shim(tmp_path, monkeypatch):
    shim = tmp_path / "claude.cmd"
    shim.write_text(NPM_SHIM, encoding="utf-8")
    exe = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setattr(f.shutil, "which", lambda name: str(shim))
    assert f.resolve_command("claude", os_name="nt") == str(exe)
    assert f.resolve_command("claude", os_name="posix") == "claude"
    exe.unlink()  # script sans exécutable : erreur claire plutôt que de passer par cmd.exe
    with pytest.raises(f.AIError, match="exécutable"):
        f.resolve_command("claude", os_name="nt")


def test_resolve_command_windows_exe_kept(tmp_path, monkeypatch):
    exe = tmp_path / "agy.exe"
    monkeypatch.setattr(f.shutil, "which", lambda name: str(exe))
    assert f.resolve_command("agy", os_name="nt") == str(exe)
    monkeypatch.setattr(f.shutil, "which", lambda name: None)
    assert f.resolve_command("absent", os_name="nt") == "absent"


def test_providers_from_env(monkeypatch):
    monkeypatch.delenv("LOLCOACH_VISION", raising=False)
    assert f.vision_from_env() is None
    monkeypatch.setenv("LOLCOACH_VISION", "inconnu")
    with pytest.raises(f.AIError):
        f.vision_from_env()
    monkeypatch.setenv("LOLCOACH_VISION", "commande")
    monkeypatch.delenv("LOLCOACH_VISION_CMD", raising=False)
    with pytest.raises(f.AIError, match="LOLCOACH_VISION_CMD"):
        f.vision_from_env()
    monkeypatch.setenv("LOLCOACH_VISION_CMD", "python -c pass {prompt}")
    assert f.vision_from_env().name == "commande"
    monkeypatch.setenv("LOLCOACH_REDACTEUR", "nimporte")
    with pytest.raises(f.AIError):
        f.writer_from_env()


def test_gemini_provider_with_fake_client(tmp_path):
    pytest.importorskip("google.genai")
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG")
    seen = {}

    class Models:
        def generate_content(self, model, contents, config):
            seen.update(model=model, contents=contents, config=config)
            return type("R", (), {"text": '{"moments": []}'})()

    client = type("C", (), {"models": Models()})()
    provider = f.GeminiProvider(client=client)
    assert provider.look("consigne", [image], VISION_SCHEMA) == {"moments": []}
    assert seen["model"] == f.DEFAULT_GEMINI_MODEL and len(seen["contents"]) == 2


def test_prompts_mention_catalogue_and_data():
    prompt = vision_prompt(Moment(600_000, "mort coûteuse"), "LeeSin", "JUNGLE", {"kda": [1, 2]})
    assert "10:00" in prompt and "JGL_GANK_SANS_PRIO" in prompt and "mort coûteuse" in prompt
    assert "OBJ_ABSENT" not in prompt  # repérée par règle, pas par l'IA
    assert '"kda"' in report_prompt({"kda": [1, 2]})
    assert "https://youtu.be/x" in cards_prompt("JUNGLE", "https://youtu.be/x")


def test_npm_shim_cannot_point_outside_its_folder(tmp_path, monkeypatch):
    shim = tmp_path / "bin" / "claude.cmd"
    shim.parent.mkdir()
    (tmp_path / "evil.exe").write_bytes(b"")
    shim.write_text('"%dp0%\\..\\evil.exe" %*\r\n', encoding="utf-8")
    monkeypatch.setattr(f.shutil, "which", lambda name: str(shim))
    with pytest.raises(f.AIError, match="exécutable"):
        f.resolve_command("claude", os_name="nt")
