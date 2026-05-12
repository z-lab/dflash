import json
import sys
import types
from types import SimpleNamespace

import pytest

import dflash.benchmark as benchmark


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload
        self.raised = False

    def raise_for_status(self):
        self.raised = True

    def json(self):
        return self.payload


def test_prepare_dataset_writes_single_and_multi_turn_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "CACHE_DIR", tmp_path)

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.load_dataset = lambda *args, **kwargs: [
        {"text": "first"},
        {"text": "second"},
    ]
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(
        benchmark.DATASETS,
        "unit-single",
        {
            "load_args": ("repo",),
            "load_kwargs": {"split": "test"},
            "format": lambda row: row["text"].upper(),
        },
    )

    path = benchmark._prepare_dataset("unit-single")

    assert path == tmp_path / "unit-single.jsonl"
    assert path.read_text().splitlines() == [
        json.dumps({"turns": ["FIRST"]}),
        json.dumps({"turns": ["SECOND"]}),
    ]

    monkeypatch.setitem(
        benchmark.DATASETS,
        "unit-multi",
        {
            "load_args": ("repo",),
            "load_kwargs": {},
            "format": lambda row: [row["text"], row["text"].upper()],
            "multi_turn": True,
        },
    )

    assert benchmark._prepare_dataset("unit-multi").read_text().splitlines()[0] == json.dumps(
        {"turns": ["first", "FIRST"]}
    )


def test_load_and_process_dataset_uses_cache_or_prepares(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "CACHE_DIR", tmp_path)
    monkeypatch.setitem(benchmark.DATASETS, "unit", {"format": lambda row: row})
    (tmp_path / "unit.jsonl").write_text('{"turns":["cached"]}\n')

    assert benchmark.load_and_process_dataset("unit") == [{"turns": ["cached"]}]

    prepared = tmp_path / "prepared.jsonl"

    def prepare(name):
        prepared.write_text('{"turns":["prepared"]}\n')
        return prepared

    monkeypatch.setattr(benchmark, "_prepare_dataset", prepare)
    monkeypatch.setitem(benchmark.DATASETS, "prepared", {"format": lambda row: row})

    assert benchmark.load_and_process_dataset("prepared") == [{"turns": ["prepared"]}]

    with pytest.raises(ValueError, match="Unknown dataset"):
        benchmark.load_and_process_dataset("missing")


def test_limit_dataset_keeps_or_shuffles_subset(monkeypatch):
    dataset = [{"i": i} for i in range(4)]

    assert benchmark._limit_dataset(dataset, None) is dataset
    assert benchmark._limit_dataset(dataset, 5) is dataset

    shuffled = []
    monkeypatch.setattr(benchmark.random, "shuffle", lambda values: shuffled.append(list(values)) or values.reverse())

    assert benchmark._limit_dataset(dataset, 2) == [{"i": 3}, {"i": 2}]
    assert shuffled == [[{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]]


def test_chat_template_decode_metrics_and_summary(capsys):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"messages": messages, **kwargs}

    assert benchmark._apply_chat_template(Tokenizer(), [{"role": "user"}], True) == {
        "messages": [{"role": "user"}],
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }

    finite = benchmark._make_decode_metrics(3, 4.0, [1, 2])
    infinite = benchmark._make_decode_metrics(0, 0, [])
    assert finite.time_per_output_token == 0.25
    assert infinite.time_per_output_token == float("inf")

    responses = [
        {
            1: SimpleNamespace(time_per_output_token=0.5),
            3: SimpleNamespace(time_per_output_token=0.25, acceptance_lengths=[1, 2]),
        },
        {
            1: SimpleNamespace(time_per_output_token=1.0),
            3: SimpleNamespace(time_per_output_token=0.5, acceptance_lengths=[2, 3]),
        },
    ]
    benchmark._print_decode_summary(responses, 3)
    out = capsys.readouterr().out
    assert "Baseline throughput:" in out
    assert "DFlash throughput:" in out
    assert "Acceptance length histogram:" in out


def test_dist_helpers(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("LOCAL_RANK", "2")

    with pytest.warns(UserWarning, match="RANK not set"):
        benchmark._dist_init(SimpleNamespace(init_process_group=lambda **kwargs: None))

    monkeypatch.setenv("RANK", "1")
    calls = []
    benchmark._dist_init(SimpleNamespace(init_process_group=lambda **kwargs: calls.append(kwargs)))
    assert calls == [{"backend": "nccl", "init_method": "env://"}]
    assert benchmark._dist_size() == 3
    assert benchmark._dist_rank() == 1
    assert benchmark._dist_local_rank() == 2
    assert benchmark._dist_is_main() is False


def test_dist_gather_all_branches(monkeypatch):
    assert benchmark._dist_gather(SimpleNamespace(is_initialized=lambda: False), "x") == ["x"]

    gathered = []

    def gather_object(obj, objs=None, dst=0):
        gathered.append((obj, objs, dst))
        if objs is not None:
            objs[0] = "rank0"

    dist = SimpleNamespace(is_initialized=lambda: True, gather_object=gather_object)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    assert benchmark._dist_gather(dist, "x", dst=2) == ["rank0"]

    monkeypatch.setenv("RANK", "1")
    assert benchmark._dist_gather(dist, "y", dst=2) is None
    assert gathered[-1] == ("y", None, 2)


def test_transformers_model_check_and_attention_impl(monkeypatch):
    benchmark._check_transformers_model("Qwen3-8B")
    benchmark._check_transformers_model("Meta-Llama-3.1-8B-Instruct")

    with pytest.raises(ValueError, match="does not support"):
        benchmark._check_transformers_model("qwen3.5")

    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    assert benchmark._get_transformers_attn_impl() == "sdpa"

    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    assert benchmark._get_transformers_attn_impl() == "flash_attention_2"


def test_send_sglang_and_vllm_payloads(monkeypatch):
    calls = []
    payloads = [DummyResponse([{"ok": "list"}]), DummyResponse({"ok": "dict"}), DummyResponse({"vllm": True})]

    def post(url, json, timeout):
        calls.append((url, json, timeout))
        return payloads.pop(0)

    monkeypatch.setattr(benchmark.requests, "post", post)

    assert benchmark._send_sglang(
        "http://host",
        "prompt",
        max_new_tokens=5,
        temperature=0.1,
        top_p=0.9,
        top_k=3,
        timeout_s=7,
    ) == {"ok": "list"}
    assert benchmark._send_sglang(
        "http://host",
        "prompt",
        max_new_tokens=5,
        temperature=0.1,
        top_p=0.9,
        top_k=3,
        timeout_s=7,
    ) == {"ok": "dict"}
    assert benchmark._send_vllm(
        "http://host",
        "prompt",
        model="model",
        max_new_tokens=5,
        temperature=0.1,
        top_p=0.9,
        top_k=3,
        timeout_s=7,
        enable_thinking=True,
    ) == {"vllm": True}

    assert calls[0] == (
        "http://host/generate",
        {
            "text": "prompt",
            "sampling_params": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 3,
                "max_new_tokens": 5,
            },
        },
        7,
    )
    assert calls[2][0] == "http://host/v1/chat/completions"
    assert calls[2][1]["chat_template_kwargs"] == {"enable_thinking": True}


def test_run_server_vllm(monkeypatch, capsys):
    args = SimpleNamespace(
        backend="vllm",
        model="model",
        dataset="unit",
        num_prompts=2,
        concurrency=1,
        base_url="http://host",
        max_new_tokens=5,
        temperature=0.1,
        top_p=0.9,
        top_k=3,
        timeout_s=7,
        enable_thinking=True,
    )
    monkeypatch.setattr(
        benchmark,
        "load_and_process_dataset",
        lambda name: [{"turns": ["a"]}, {"turns": ["b"]}],
    )
    sent = []

    def send_vllm(base_url, text, **kwargs):
        sent.append((base_url, text, kwargs))
        return {"usage": {"completion_tokens": 4}}

    monkeypatch.setattr(benchmark, "_send_vllm", send_vllm)
    benchmark._run_server(args)

    out = capsys.readouterr().out
    assert len(sent) == 3
    assert "Backend:          vllm" in out
    assert "Output tokens:    8" in out


def test_run_server_sglang_flush_warning_and_metrics(monkeypatch, capsys):
    args = SimpleNamespace(
        backend="sglang",
        model="model",
        dataset="unit",
        num_prompts=1,
        concurrency=1,
        base_url="http://host",
        max_new_tokens=5,
        temperature=0.1,
        top_p=0.9,
        top_k=3,
        timeout_s=7,
        enable_thinking=False,
    )
    monkeypatch.setattr(benchmark, "load_and_process_dataset", lambda name: [{"turns": ["hello"]}])

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return f"templated:{messages[0]['content']}"

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(benchmark.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))
    args.num_prompts = 2
    accept_lengths = iter(["ignored-warmup", "bad", "1.5"])
    monkeypatch.setattr(
        benchmark,
        "_send_sglang",
        lambda *args, **kwargs: {
            "meta_info": {
                "completion_tokens": "3",
                "spec_verify_ct": "2",
                "spec_accept_length": next(accept_lengths),
            }
        },
    )

    benchmark._run_server(args)

    out = capsys.readouterr().out
    assert "Warning: /flush_cache failed. Continuing." in out
    assert "Accept length:    1.500" in out
    assert "Spec verify ct:   4" in out


def test_main_dispatch_and_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "_run_server", lambda args: calls.append(("server", args.backend)))
    monkeypatch.setattr(benchmark, "_run_transformers", lambda args: calls.append(("transformers", args.draft_model)))
    monkeypatch.setattr(benchmark, "_run_mlx", lambda args: calls.append(("mlx", args.draft_model)))

    monkeypatch.setattr(sys, "argv", ["prog", "--backend", "vllm", "--model", "m", "--dataset", "d"])
    benchmark.main()
    assert calls[-1] == ("server", "vllm")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--backend", "transformers", "--model", "m", "--dataset", "d", "--draft-model", "draft"],
    )
    benchmark.main()
    assert calls[-1] == ("transformers", "draft")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--backend", "mlx", "--model", "m", "--dataset", "d", "--draft-model", "draft"],
    )
    benchmark.main()
    assert calls[-1] == ("mlx", "draft")

    monkeypatch.setattr(sys, "argv", ["prog", "--backend", "mlx", "--model", "m", "--dataset", "d"])
    with pytest.raises(SystemExit):
        benchmark.main()

    monkeypatch.setattr(sys, "argv", ["prog", "--backend", "transformers", "--model", "m", "--dataset", "d"])
    with pytest.raises(SystemExit):
        benchmark.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--backend", "vllm", "--model", "qwen3-4b", "--dataset", "d", "--enable-thinking"],
    )
    with pytest.raises(AssertionError, match="not trained with thinking traces"):
        benchmark.main()
