from __future__ import annotations

"""Einstiegspunkt für die Analyse-Pipeline."""

import argparse
import logging
import os
import sys
from rubin.settings import load_config
from rubin.pipelines.analysis_pipeline import AnalysisPipeline


class _ProgressOnlyStdout:
    """Schreibt nur [rubin]-Zeilen auf den gespeicherten Pipe-FD.
    Alles andere wird verworfen (Python-Level).
    
    Beachtet dass print() write(text) und write('\\n') SEPARAT aufruft.
    Nach einem [rubin]-Write wird der nächste Write (Newline) durchgelassen."""

    def __init__(self, pipe_fd_writer):
        self._pipe = pipe_fd_writer
        self._pass_next = False

    def write(self, s):
        if "[rubin]" in s:
            self._pipe.write(s)
            self._pass_next = True  # Nächsten write (Newline) auch durchlassen
        elif self._pass_next:
            self._pipe.write(s)
            self._pipe.flush()
            self._pass_next = False
        return len(s)

    def flush(self):
        self._pipe.flush()

    def fileno(self):
        return self._pipe.fileno()


def _setup_quiet_mode():
    """Leitet stdout auf OS-Level nach /dev/null um.

    KRITISCH für Performance: LightGBM's C++ Engine schreibt direkt auf
    File Descriptor 1 (stdout), bypassed Python's sys.stdout komplett.
    Ein Python-Level Filter fängt das NICHT ab.

    Lösung:
    1. Originalen stdout-FD (Pipe zum Server) sichern
    2. FD 1 → /dev/null umbiegen (fängt ALLE stdout-Writes, auch C++)
    3. sys.stdout → Wrapper der [rubin]-Zeilen auf den gesicherten FD schreibt
    """
    # Original-Pipe-FD sichern (der FD, den der Server liest)
    orig_fd = os.dup(sys.stdout.fileno())
    pipe_writer = os.fdopen(orig_fd, "w")

    # FD 1 → /dev/null: Fängt LightGBM C++, EconML, sklearn, etc.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)

    # sys.stdout: Nur [rubin]-Progress-Lines → gesicherter Pipe-FD
    sys.stdout = _ProgressOnlyStdout(pipe_writer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yml")
    ap.add_argument(
        "--quiet", action="store_true",
        help="Unterdrückt stdout außer [rubin]-Progress (für App-Modus).",
    )
    ap.add_argument(
        "--export-bundle",
        action="store_true",
        default=None,
        help="Überschreibt bundle.enabled aus der Konfiguration und exportiert am Ende ein Bundle.",
    )
    ap.add_argument(
        "--no-export-bundle",
        dest="export_bundle",
        action="store_false",
        help="Überschreibt bundle.enabled aus der Konfiguration und deaktiviert den Bundle-Export.",
    )
    ap.add_argument(
        "--bundle-dir",
        default=None,
        help="Überschreibt bundle.base_dir aus der Konfiguration.",
    )
    ap.add_argument(
        "--bundle-id",
        default=None,
        help="Überschreibt bundle.bundle_id aus der Konfiguration.",
    )
    args = ap.parse_args()

    # App-Modus oder RUBIN_QUIET env: Stdout auf OS-Level nach /dev/null
    if args.quiet or os.environ.get("RUBIN_QUIET") == "1":
        _setup_quiet_mode()

    # Logging auf stderr konfigurieren. Im App-Modus (--quiet) wird stdout
    # umgeleitet, aber stderr bleibt offen → Server fängt die Logs ab und
    # liefert sie als stderr_tail im Progress-Endpoint an die UI.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    cfg = load_config(args.config)
    pipe = AnalysisPipeline(cfg)
    pipe.run(export_bundle=args.export_bundle, bundle_dir=args.bundle_dir, bundle_id=args.bundle_id)


if __name__ == "__main__":
    main()
