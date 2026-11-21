import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from dflash.cli import main
from dflash.edit_benchmark import CASES, MODULES, SOURCE, fenced, grade


class EditBenchTest(unittest.TestCase):
    def test_complete_file(self):
        answer = SOURCE.replace("feature_043 = 43\n", "feature_043 = 143\n")
        self.assertTrue(grade("single-file", fenced(answer, "python")))
        self.assertFalse(grade("single-file", answer.rstrip()))
        self.assertFalse(grade("single-file", SOURCE))

    def test_patch_application(self):
        blocks = [f"{name}\n<<<<<<< SEARCH\n{before}=======\n{before.replace('2', '4')}>>>>>>> REPLACE\n"
                  for name, before in MODULES.items()]
        self.assertTrue(grade("multi-file", "\n".join(blocks)))
        self.assertFalse(grade("multi-file", blocks[0]))
        self.assertFalse(grade("multi-file", blocks[0] + blocks[0] + blocks[1]))
        self.assertFalse(grade("multi-file", "\n".join(blocks).replace("policy.py", "../policy.py")))

    def test_sql(self):
        self.assertTrue(grade("new-code", "SELECT sensor, SUM(value) AS t FROM readings GROUP BY sensor "
                              "HAVING t > 0 ORDER BY t DESC, sensor ASC"))
        self.assertFalse(grade("new-code", "DELETE FROM readings"))
        self.assertFalse(grade("new-code", "SELECT load_extension('unused')"))
        self.assertFalse(grade("new-code", "SELECT sensor, SUM(value) FROM readings GROUP BY sensor"))

    def test_cli_report_excludes_warmup_and_rejects_truncation(self):
        prompts = []

        def respond(url, *, json, timeout):
            self.assertEqual(url, "http://example.invalid/v1/chat/completions")
            self.assertFalse(json["chat_template_kwargs"]["enable_thinking"])
            prompts.append(json["messages"][0]["content"])
            response = Mock()
            response.json.return_value = {
                "choices": [{"finish_reason": "length", "message": {"content": ""}}],
                "usage": {"completion_tokens": 5},
            }
            return response

        with patch("dflash.benchmark.requests.post", side_effect=respond), redirect_stdout(io.StringIO()) as out:
            main(["benchmark-edits", "--base-url", "http://example.invalid", "--model", "target",
                  "--label", "stock", "--repeats", "2", "--reasoning", "off"])

        report = json.loads(out.getvalue())
        self.assertEqual(prompts[1:], list(CASES.values()) * 2)
        self.assertEqual(len(prompts), 7)
        self.assertEqual(report["label"], "stock")
        self.assertEqual([r["case"] for r in report["requests"]], list(CASES) * 2)
        for row in report["requests"]:
            self.assertFalse(row["passed"])
            self.assertEqual(row["output_tokens"], 5)
            self.assertIsNone(row["server_decode_tps"])
