#!/usr/bin/env python3
"""Score your hybrid-replay decisions: did your takes beat the raw signals, and which reasons predicted anything?
  python scripts/replay_report.py [--session ID] [--out reports/replay_report.md]
Only outcomes the replay clock has passed are counted, unless the session was finished."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.replay import report
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--session"); p.add_argument("--out")
a = p.parse_args()
md = report.to_markdown(report.analyse([a.session] if a.session else None))
if a.out:
    Path(a.out).write_text(md); print(f"wrote {a.out}")
else:
    print(md)
