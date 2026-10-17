"""Tune, benchmark and save all models.

    python scripts/train_models.py --mode quick     # fast development run
    python scripts/train_models.py --mode full      # larger search (final results)
    python scripts/train_models.py --force          # retrain even if nothing changed
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from climate_yield.benchmark import run_benchmark
from climate_yield.config import load_config
from climate_yield.pipeline import fingerprint


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["quick", "full"], default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    mode = args.mode or cfg["training"]["mode"]
    fp = fingerprint(cfg, {"mode": mode})
    fp_file = cfg.path("models_dir") / "fingerprint.json"
    if not args.force and fp_file.exists() and json.loads(fp_file.read_text())["fingerprint"] == fp:
        print("Models are up to date with data, config, code and mode; use --force to retrain.")
        return 0
    t0 = time.perf_counter()
    summary = run_benchmark(cfg, mode)
    fp_file.write_text(json.dumps({"fingerprint": fp, "mode": mode,
                                   "seconds": round(time.perf_counter() - t0, 1)}, indent=2))
    print(f"done in {time.perf_counter() - t0:.0f}s; best model: {summary['best_model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
