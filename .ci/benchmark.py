import sys
from src.benchmark.utils import read_metrics, to_html_table

if __name__ == "__main__":
    # Generate statistics report
    metrics = read_metrics(sys.argv[1], metric="accuracy")
    html_table = to_html_table(metrics)
    print(html_table)
