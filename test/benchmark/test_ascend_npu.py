import unittest
import src.benchmark.utils as utils

path = "ascend_npu_benchmark.json"


class TestBenchmark(unittest.TestCase):
    def test_read_metrics(self):
        metrics = utils.read_metrics(path, metric="accuracy")
        self.assertTrue(len(metrics) == 2)
        for metric in metrics:
            self.assertEqual(metric.key.device, "npu")
            self.assertEqual(metric.value, "pass")

    def test_to_html_table(self):
        metrics = utils.read_metrics(path, metric="accuracy")
        html_table = utils.to_markdown_table(metrics)
        self.assertIsNotNone(html_table)


if __name__ == "__main__":
    unittest.main()
