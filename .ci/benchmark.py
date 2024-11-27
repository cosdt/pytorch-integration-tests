import os
import sys
from src.benchmark.utils import read_metrics, to_html_table

if __name__ == "__main__":
    # Generate statistics report
    statistics_path = sys.argv[1]
    metrics = read_metrics(statistics_path, metric="accuracy")
    html_table = to_html_table(metrics)

    # Write to workflow job summary
    summary_path = os.environ["GITHUB_STEP_SUMMARY"]
    with open(summary_path, "a") as f:
        f.write("## Torchbenchmark statistics report\n")
        f.write(html_table)
