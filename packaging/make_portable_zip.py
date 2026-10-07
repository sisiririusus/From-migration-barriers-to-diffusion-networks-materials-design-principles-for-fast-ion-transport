"""Create a checked ZIP and per-file SHA256 manifest for a portable build."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PLATFORM = "Windows" if sys.platform == "win32" else "Linux"
APP = ROOT / "build" / PLATFORM / "MigrationWorkbench"
OUT = ROOT / "release-assets" / ("MigrationWorkbench_0.2_%s_x86_64.zip" % PLATFORM)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    executable = APP / ("MigrationWorkbench.exe" if PLATFORM == "Windows" else "MigrationWorkbench")
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if OUT.exists():
        raise FileExistsError("Existing release is preserved: %s" % OUT)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted((path for path in APP.rglob("*") if path.is_file() and path.name != "release_manifest.json"), key=lambda path: str(path.relative_to(APP)).lower())
    manifest = {
        "product": "Migration Network Workbench",
        "version": "0.2",
        "platform": PLATFORM + " x86-64",
        "files": [{"path": path.relative_to(APP).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
    }
    manifest_path = APP / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(str(OUT), "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in files + [manifest_path]:
            archive.write(str(path), arcname="MigrationWorkbench/" + path.relative_to(APP).as_posix())
    with zipfile.ZipFile(str(OUT)) as archive:
        bad = archive.testzip()
        if bad:
            raise IOError("ZIP CRC failed: %s" % bad)
    print("RELEASE=" + str(OUT))
    print("FILES=%d ZIP_MIB=%.2f SHA256=%s" % (len(files) + 1, OUT.stat().st_size / 1024**2, sha256(OUT)))


if __name__ == "__main__":
    main()
