"""Exercise the real JSONL wrapper without loading a model or resident service."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class LauncherEncodingTests(unittest.TestCase):
    def test_utf8_names_survive_a_non_utf8_windows_launcher_locale(self):
        plugin = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(plugin / "bot.py", root / "bot.py")
            module = root / "src/rin/inference/native"
            module.mkdir(parents=True)
            for package in (module, module.parent, module.parent.parent):
                (package / "__init__.py").touch()
            (module / "client.py").write_text(
                "import json,sys\n"
                "for line in sys.stdin:\n"
                "    event=json.loads(line)[0]\n"
                "    print(json.dumps({'type':'none','names':event['names']},ensure_ascii=False),flush=True)\n",
                encoding="utf-8",
            )
            names = ["東風", "小鳥遊", "🀄", "Tester"]
            payload = json.dumps([{"type": "start_game", "id": 0, "names": names}], ensure_ascii=False) + "\n"
            env = os.environ.copy()
            env.pop("AKAGI_BOT_CONFIG", None)
            env["RIN_NATIVE_PYTHON"] = sys.executable
            env["PYTHONUTF8"] = "0"
            env["PYTHONIOENCODING"] = "cp1252:surrogateescape"
            result = subprocess.run([sys.executable, str(root / "bot.py"), "0"],
                                    input=payload, text=True, encoding="utf-8",
                                    capture_output=True, env=env, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"type": "none", "names": names})
            self.assertEqual(len(result.stdout.splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
