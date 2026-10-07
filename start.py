"""Start the CLI or desktop UI with the audited local Python environment."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import ctypes


def main():
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    extra = [
        root / "packaging" / "vendor_py37",
        root / "src",
        root / "science" / "repro" / "scripts" / "reproduction" / "project" / "scripts",
    ]
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(path) for path in extra] + ([existing] if existing else []))
    for path in reversed(extra):
        sys.path.insert(0, str(path))
    if sys.argv[1:2] == ["--cli"]:
        from migration_workbench.cli import main as cli_main

        return cli_main(sys.argv[2:])
    qt_bin = root / "packaging" / "vendor_py37" / "PyQt5" / "Qt5" / "bin"
    if qt_bin.is_dir():
        os.environ["PATH"] = str(qt_bin) + os.pathsep + os.environ.get("PATH", "")
        # The legacy Conda installation also contains Qt 5.9. Load the wheel's
        # Qt 5.15 libraries by absolute path so Windows cannot select that ABI.
        for name in ("Qt5Core.dll", "Qt5Gui.dll", "Qt5Widgets.dll"):
            ctypes.WinDLL(str(qt_bin / name))
    from migration_workbench.gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
