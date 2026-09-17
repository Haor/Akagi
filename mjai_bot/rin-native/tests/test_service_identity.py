"""A resident predictor must never survive a changed model or input adapter."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rin.inference.native import service
from rin.inference.native.settings import Settings


class IdentityTests(unittest.TestCase):
    def test_pool_key_tracks_model_config_code_and_native_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "models/ppo400"
            source = root / "src/rin/inference/native/service.py"
            model.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            tracked = [model / name for name in
                       ("actor.safetensors", "config.json", "conversion-manifest.json")]
            tracked += [source, root / "libriichi.pyd"]
            for path in tracked:
                path.write_bytes(b"original")
            with patch.object(service, "__file__", str(source)), patch(
                    "importlib.util.find_spec", return_value=SimpleNamespace(origin=str(tracked[-1]))):
                original = service.endpoint_path(root, Settings())
                self.assertEqual(original, service.endpoint_path(root, Settings()))
                for path in tracked:
                    path.write_bytes(b"replacement")
                    self.assertNotEqual(original, service.endpoint_path(root, Settings()), str(path))
                    path.write_bytes(b"original")
                for name in ("observer.safetensors", "observer-config.json", "observer-manifest.json"):
                    observer_file = model / name
                    observer_file.write_bytes(b"observer")
                    self.assertNotEqual(original, service.endpoint_path(root, Settings()), name)
                    observer_file.unlink()
                self.assertNotEqual(original, service.endpoint_path(root, Settings(cpu_threads=2)))


if __name__ == "__main__":
    unittest.main()
