"""Summarize and plot per-rank training timing CSVs.

Usage: python analyze_debug_timing.py /path/to/logs
"""

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def find_csv_files(inputs):
    files = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_dir():
            files.extend(sorted(path.glob("debug_timing_rank_*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
        else:
            raise ValueError(f"Not a CSV file or directory: {path}")

    unique_files = list(dict.fromkeys(path.resolve() for path in files))
    if not unique_files:
        raise ValueError("No debug_timing_rank_*.csv files found in the inputs.")
    return unique_files


def load_rows(csv_files):
    rows = []
    for path in csv_files:
        match = re.search(r"rank_(\d+)", path.stem)
        rank = match.group(1) if match else path.stem
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required = {"step", "stage", "seconds"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{path} must have columns: step, stage, seconds"
                )
            for line_number, row in enumerate(reader, start=2):
                if not row.get("step") or not row.get("stage") or not row.get("seconds"):
                    continue
                try:
                    step = int(row["step"])
                    seconds = float(row["seconds"])
                except ValueError as error:
                    raise ValueError(f"Invalid timing value in {path}:{line_number}") from error
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError(f"Invalid duration in {path}:{line_number}: {seconds}")
                rows.append({
                    "rank": rank,
                    "step": step,
                    "stage": row["stage"],
                    "seconds": seconds,
                })

    if not rows:
        raise ValueError("The timing CSV files contain no data rows.")
    return rows


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["rank"], row["stage"])].append(row["seconds"])

    summaries = []
    for (rank, stage), values in sorted(groups.items()):
        summaries.append({
            "rank": rank,
            "stage": stage,
            "count": len(values),
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "p95_seconds": percentile(values, 0.95),
            "min_seconds": min(values),
            "max_seconds": max(values),
        })
    return summaries


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed; skipping PNG plot.", file=sys.stderr)
        return False

    by_step_stage = defaultdict(list)
    full_step_by_rank = defaultdict(list)
    for row in rows:
        by_step_stage[(row["stage"], row["step"])].append(row["seconds"])
        if row["stage"] == "full_step":
            full_step_by_rank[row["rank"]].append((row["step"], row["seconds"]))

    stages = sorted({row["stage"] for row in rows if row["stage"] != "full_step"})
    figure, (stage_axis, rank_axis) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for stage in stages:
        points = sorted(
            (step, values)
            for (point_stage, step), values in by_step_stage.items()
            if point_stage == stage
        )
        steps = [step for step, _ in points]
        medians = [statistics.median(values) for _, values in points]
        lows = [min(values) for _, values in points]
        highs = [max(values) for _, values in points]
        stage_axis.plot(steps, medians, label=stage, linewidth=1.2)
        if len({len(values) for _, values in points}) == 1 and points and len(points[0][1]) > 1:
            stage_axis.fill_between(steps, lows, highs, alpha=0.12)

    stage_axis.set_ylabel("Seconds (median across ranks)")
    stage_axis.set_title("Stage timings per training step")
    stage_axis.grid(True, alpha=0.25)
    stage_axis.legend(loc="upper left", ncol=2, fontsize="small")

    rank_ids = sorted({row["rank"] for row in rows})
    for rank in rank_ids:
        points = sorted(full_step_by_rank[rank])
        if points:
            rank_axis.plot(
                [step for step, _ in points],
                [seconds for _, seconds in points],
                label=f"rank {rank}",
                linewidth=1,
            )

    rank_axis.set_xlabel("Training step")
    rank_axis.set_ylabel("Full step seconds")
    rank_axis.set_title("Full-step timing by rank")
    rank_axis.grid(True, alpha=0.25)
    rank_axis.legend(loc="upper left", ncol=4, fontsize="small")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Log directory or one or more debug_timing_rank_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (defaults to the first input directory, or the CSV's parent)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write CSV analysis only",
    )
    args = parser.parse_args()

    try:
        csv_files = find_csv_files(args.inputs)
        rows = load_rows(csv_files)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    default_output = Path(args.inputs[0])
    if not default_output.is_dir():
        default_output = default_output.parent
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "timing_combined.csv"
    summary_path = output_dir / "timing_summary.csv"
    write_csv(combined_path, rows, ["rank", "step", "stage", "seconds"])
    summaries = summarize(rows)
    write_csv(
        summary_path,
        summaries,
        [
            "rank", "stage", "count", "mean_seconds", "median_seconds",
            "p95_seconds", "min_seconds", "max_seconds",
        ],
    )

    print(f"Loaded {len(rows)} timing rows from {len(csv_files)} file(s).")
    print(f"Combined rows: {combined_path}")
    print(f"Rank/stage summary: {summary_path}")
    if not args.no_plot:
        plot_path = output_dir / "timing_plot.png"
        if plot_rows(rows, plot_path):
            print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()