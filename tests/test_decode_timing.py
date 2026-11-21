import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from dflash.benchmark import _print_decode_summary
from dflash.model import dflash_generate


class DecodeTimingTest(unittest.TestCase):
    def generate(self, count, stop_token_ids=None):
        draft = SimpleNamespace(block_size=1, config=None, mask_token_id=0)
        target = Mock(device=torch.device("cpu"), config=None)
        target.side_effect = lambda ids, **kwargs: SimpleNamespace(
            logits=torch.ones(1, ids.shape[1], 1)
        )
        with (
            patch("dflash.model._make_cache", return_value=Mock(get_seq_length=lambda: 0)),
            patch("dflash.model._cuda_time", side_effect=[0., 2., 2., 5.]),
        ):
            return dflash_generate(
                draft, target, torch.tensor([[0]]), count, stop_token_ids,
                return_stats=True,
            )

    def test_decode_excludes_prefill_token(self):
        response = self.generate(4)
        self.assertEqual(response.num_output_tokens, 4)
        self.assertEqual(response.time_to_first_token, 2.)
        self.assertEqual(response.time_per_output_token, 1.)

    def test_cap_or_eos_during_prefill_has_no_decode_rate(self):
        for limit, stop in [(1, None), (4, [0])]:
            with self.subTest(limit=limit, stop=stop):
                response = self.generate(limit, stop)
                self.assertEqual(response.num_output_tokens, 1)
                self.assertIsNone(response.time_per_output_token)

    def test_summary_excludes_only_unmeasurable_pairs(self):
        single, full = self.generate(1), self.generate(4)
        with redirect_stdout(io.StringIO()) as out:
            _print_decode_summary([{1: single, 8: single}], 8)
        self.assertIn("n/a", out.getvalue())
        with redirect_stdout(io.StringIO()) as out:
            _print_decode_summary([{1: full, 8: single}, {1: full, 8: full}], 8)
        self.assertIn("Decoding speedup: 1.00", out.getvalue())
