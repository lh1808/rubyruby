"""Runner für die DataPrepPipeline.
Die Datenaufbereitung wird über die zentrale Projekt-Konfiguration gesteuert.
Voraussetzung ist eine Sektion `data_prep` in der verwendeten YAML-Datei.
Beispiel:
python run_dataprep.py --config configs/config_full_example.yml
Der Lauf erzeugt die Artefakte (X/T/Y + Preprocessing) in `data_prep.output_path`.
Diese Pfade werden anschließend in `data_files` der gleichen Konfiguration referenziert
und von der Analyse-Pipeline genutzt."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from rubin.pipelines.data_prep_pipeline import DataPrepPipeline


class _ProgressOnlyStdout:
    """Schreibt nur [rubin]-Zeilen auf den gespeicherten Pipe-FD.
    print() ruft write(text) und write('\\n') SEPARAT auf."""
    def __init__(self, pipe_fd_writer):
        self._pipe = pipe_fd_writer
        self._pass_next = False
    def write(self, s):
        if "[rubin]" in s:
            self._pipe.write(s)
            self._pass_next = True
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
    """Leitet stdout auf OS-Level nach /dev/null um (fängt auch C++ Output)."""
    orig_fd = os.dup(sys.stdout.fileno())
    pipe_writer = os.fdopen(orig_fd, "w")
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    sys.stdout = _ProgressOnlyStdout(pipe_writer)


def main() -> None:
    parser = argparse.ArgumentParser(description="DataPrep für rubin ausführen")
    parser.add_argument("--config", required=True, help="Pfad zur zentralen Konfiguration (YAML)")
    parser.add_argument("--quiet", action="store_true", help="Nur [rubin]-Progress auf stdout")
    args = parser.parse_args()

    if args.quiet or os.environ.get("RUBIN_QUIET") == "1":
        _setup_quiet_mode()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    pipeline = DataPrepPipeline.from_config_path(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
