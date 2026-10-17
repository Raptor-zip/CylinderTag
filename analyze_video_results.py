"""Summarize detector CSV runs, without additional Python dependencies."""
import csv
import json
from collections import Counter
from pathlib import Path
import sys


def summarize(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No frames in {path}")
    hits = [row for row in rows if int(row["id0_detected"])]
    ids = Counter()
    other_frames = 0
    for row in rows:
        values = [int(x) for x in row["ids"].split(";") if x]
        ids.update(set(values))
        other_frames += any(x != 0 for x in values)
    return {
        "run": path.stem,
        "frames": len(rows),
        "id0_frames": len(hits),
        "id0_percent": 100 * len(hits) / len(rows),
        "any_id_frames": sum(int(r["marker_count"]) > 0 for r in rows),
        "other_id_frames": other_frames,
        "frames_per_id": dict(sorted(ids.items())),
        "id0_events": [
            {"frame": int(row["frame"]), "time_seconds": float(row["time_seconds"]),
             "columns": int(row["id0_columns"])} for row in hits
        ],
        "average_detector_ms": sum(float(r["processing_ms"]) for r in rows) / len(rows),
    }


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    results = [summarize(path) for path in sorted(folder.glob("*.csv"))]
    if not results:
        raise SystemExit("No CSV files found")
    report = {"expected_id": 0, "note": "ID outputs are not calibrated pose accuracy measurements. "
              "Configuration comparisons use the same clip and are exploratory.", "runs": results}
    (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for run in results:
        print(f"{run['run']}: ID 0 {run['id0_frames']}/{run['frames']} "
              f"({run['id0_percent']:.2f}%), other-ID frames {run['other_id_frames']}, "
              f"events={run['id0_events']}")
