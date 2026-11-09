"""Trust-chain and extraction failures must precede installation/provider loading."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
spec = importlib.util.spec_from_file_location("native_materializer", TOOLS / "materialize_native_providers.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class NativePackageTests(unittest.TestCase):
    def archive(self, root, name="packages/vendors/test/op_api/lib/provider.so"):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as package:
            info = tarfile.TarInfo(name)
            info.size = 3
            package.addfile(info, io.BytesIO(b"api"))
        path = root / "package.run"
        path.write_bytes(b'#!/bin/sh\nskip="3"\nexit 97\n' + payload.getvalue())
        return path, {"package_bytes": path.stat().st_size,
                      "package_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_verified_payload_extracts_without_running_shell(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, spec = self.archive(root)
            m.extract_verified_archive(archive, spec, root / "out")
            self.assertEqual((root / "out/packages/vendors/test/op_api/lib/provider.so").read_bytes(), b"api")

    def test_wrong_package_hash_prevents_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, spec = self.archive(root)
            spec["package_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                m.extract_verified_archive(archive, spec, root / "out")
            self.assertFalse((root / "out").exists())

    def test_traversal_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, spec = self.archive(root, "../escaped")
            with self.assertRaises(tarfile.FilterError):
                m.extract_verified_archive(archive, spec, root / "out")
            self.assertFalse((root / "escaped").exists())

    def test_missing_and_changed_provider_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {"files": {"provider.so": hashlib.sha256(b"api").hexdigest()}}
            with self.assertRaisesRegex(RuntimeError, "dependency mismatch"):
                m.verify(root, spec)
            (root / "provider.so").write_bytes(b"wrong provider")
            with self.assertRaisesRegex(RuntimeError, "dependency mismatch"):
                m.verify(root, spec)
            (root / "provider.so").write_bytes(b"api")
            m.verify(root, spec)


if __name__ == "__main__":
    unittest.main()
