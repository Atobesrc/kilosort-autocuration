"""Command line interface: ks-autocurate run <kilosort_folder> [options]."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ks-autocurate",
        description="Automated post-Kilosort curation: merge over-splits, relabel units, "
                    "and re-export in Kilosort/phy format.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="curate one Kilosort output folder")
    run_p.add_argument("ks_folder", help="Kilosort output folder (read-only input)")
    run_p.add_argument("--recording", default=None,
                       help="raw binary file; default taken from params.py dat_path")
    run_p.add_argument("--out", default=None,
                       help="output folder; default '<ks_folder>_curated' next to the input")
    run_p.add_argument("--work-dir", default=None,
                       help="cache folder for the SortingAnalyzer; default '<out>_work'")
    run_p.add_argument("--config", default=None, help="YAML overriding configs/default.yaml")
    run_p.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        from .pipeline import run
        out = run(args.ks_folder, recording_path=args.recording, out_folder=args.out,
                  work_dir=args.work_dir, config_path=args.config, verbose=not args.quiet)
        print(out)


if __name__ == "__main__":
    main()
