import ast
import dataclasses
import json
import pathlib
from jinja2 import Template
from typing import Dict, List, Optional


# Same as torchbenchmark.util.experiment.instantiator.TorchBenchModelConfig
# https://github.com/pytorch/benchmark/blob/main/torchbenchmark/util/experiment/instantiator.py#L26
@dataclasses.dataclass
class TorchBenchModelConfig:
    name: str
    test: str
    device: str
    batch_size: Optional[int]
    extra_args: List[str]
    extra_env: Optional[Dict[str, str]] = None
    output_dir: Optional[pathlib.Path] = None


@dataclasses.dataclass
class TorchBenchModelMetric:
    key: TorchBenchModelConfig
    value: str


def read_json(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
        return data


def save_file(path: str, data) -> None:
    with open(path, 'w') as file:
        file.write(data)


def parse_to_dict(config_str: str):
    """
    Parse a string (like 'key1=value1, key2=value2, ...') into a dict
    """
    items = config_str.split(", ")
    config = {}

    for item in items:
        key, value = item.split("=", 1)
        try:
            config[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            config[key] = value

    return config


def read_metrics(path: str, *, metric=None) -> List[TorchBenchModelMetric]:
    output = read_json(path)
    metrics_data = output.get('metrics', {})

    metrics = []
    for metric_key, metric_value in metrics_data.items():
        key_dict = parse_to_dict(metric_key)
        if metric is None or metric == key_dict["metric"]:
            config = TorchBenchModelConfig(
                name=key_dict.get("model"),
                test=key_dict.get("test"),
                device=key_dict.get("device"),
                batch_size=key_dict.get("batch_size"),
                extra_args=key_dict.get("extra_args"),
                extra_env=key_dict.get("extra_env"),
                output_dir=key_dict.get("output_dir"),
            )
            model_metric = TorchBenchModelMetric(config, metric_value)
            metrics.append(model_metric)
    return metrics


def generate_table(data):
    template = Template("""
    <table border="1">
        {% for row in data %}
        <tr>
            {% for cell in row %}
            <td>{{ cell }}</td>
            {% endfor %}
        </tr>
        {% endfor %}
    </table>
    """)
    return template.render(data=data)


def to_html_table(metrics: List[TorchBenchModelMetric]):
    models = list({metric.key.name for metric in metrics})
    devices = list({metric.key.device for metric in metrics})

    def filter_result(metrics: List[TorchBenchModelMetric], *, model, device):
        for metric in metrics:
            if metric.key.name == model and metric.key.device == device:
                return metric

    rows = []
    for model in models:
        row = [model]
        for device in devices:
            metric = filter_result(metrics, model=model, device=device)
            if metric is not None:
                is_pass = metric.value == "pass"
                cell = "✅" if is_pass else "❌"
            else:
                cell = ""
            row.append(cell)
        rows.append(row)

    header = [""] + devices
    data = [header, *rows]
    html_table = generate_table(data)
    return html_table
