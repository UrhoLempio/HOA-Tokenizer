"""Summarize per-second GPU utilization CSVs from a profiling directory.

Usage: python analyze_gpu_utilization.py profiling_<jobid>
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_samples(directory):
    files = sorted(directory.glob("gpu_utilization_*.csv"))
    if not files:
        raise ValueError(f"No gpu_utilization_*.csv files found in {directory}")

    samples = defaultdict(list)
    for path in files:
        node = path.stem.removeprefix("gpu_utilization_")
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required = {
                "timestamp", "index", "uuid", "gpu_util_pct", "mem_util_pct",
                "mem_used_mib", "mem_total_mib", "power_w",
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected GPU CSV columns in {path}")

            for line_number, row in enumerate(reader, start=2):
                try:
                    gpu_util = float(row["gpu_util_pct"].strip())
                    values = {
                        "gpu_util_pct": gpu_util,
                        "mem_util_pct": float(row["mem_util_pct"].strip()),
                        "mem_used_mib": float(row["mem_used_mib"].strip()),
                        "mem_total_mib": float(row["mem_total_mib"].strip()),
                        "power_w": float(row["power_w"].strip()),
                    }
                except (AttributeError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid sample in {path}:{line_number}") from error
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError(f"Non-finite sample in {path}:{line_number}")
                values.update({
                    "node": node,
                    "index": row["index"].strip(),
                    "uuid": row["uuid"].strip(),
                    "timestamp": row["timestamp"].strip(),
                })
                samples[(node, values["index"], values["uuid"])].append(values)

    if not samples:
        raise ValueError("The GPU CSV files contain no samples.")
    return samples


def summarize(samples):
    summaries = []
    for (node, index, uuid), records in sorted(samples.items()):
        gpu_util = [record["gpu_util_pct"] for record in records]
        mem_util = [record["mem_util_pct"] for record in records]
        mem_used = [record["mem_used_mib"] for record in records]
        power = [record["power_w"] for record in records]
        summaries.append({
            "node": node,
            "gpu_index": index,
            "gpu_uuid": uuid,
            "samples": len(records),
            "gpu_util_mean_pct": statistics.mean(gpu_util),
            "gpu_util_median_pct": statistics.median(gpu_util),
            "gpu_util_p95_pct": percentile(gpu_util, 0.95),
            "samples_gpu_below_10_pct": 100 * sum(value < 10 for value in gpu_util) / len(gpu_util),
            "samples_gpu_at_least_90_pct": 100 * sum(value >= 90 for value in gpu_util) / len(gpu_util),
            "mem_util_mean_pct": statistics.mean(mem_util),
            "mem_used_mean_mib": statistics.mean(mem_used),
            "power_mean_w": statistics.mean(power),
        })
    return summaries


def write_summary(path, summaries):
    fields = list(summaries[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path, help="Directory containing GPU utilization CSVs")
    parser.add_argument("--output", type=Path, help="Summary CSV path (defaults inside profile_dir)")
    args = parser.parse_args()

    try:
        samples = load_samples(args.profile_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    summaries = summarize(samples)
    output_path = args.output or args.profile_dir / "gpu_utilization_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary(output_path, summaries)

    print(f"Analyzed {sum(row['samples'] for row in summaries)} GPU samples across {len(summaries)} GPUs.")
    print(f"Summary: {output_path}")
    print("node GPU samples mean% median% p95% <10% >=90% mean_power_W")
    for row in summaries:
        print(
            f"{row['node']} {row['gpu_index']} {row['samples']} "
            f"{row['gpu_util_mean_pct']:.1f} {row['gpu_util_median_pct']:.1f} "
            f"{row['gpu_util_p95_pct']:.1f} {row['samples_gpu_below_10_pct']:.1f} "
            f"{row['samples_gpu_at_least_90_pct']:.1f} {row['power_mean_w']:.1f}"
        )


if __name__ == "__main__":
    main()