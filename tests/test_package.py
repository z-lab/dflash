import sys
import types

import pytest

import dflash


def test_lazy_dataset_export():
    from dflash.benchmark import load_and_process_dataset

    assert dflash.load_and_process_dataset is load_and_process_dataset


def test_lazy_model_exports_from_loaded_module(monkeypatch):
    sample = object()
    model_mod = types.ModuleType("dflash.model")
    model_mod.DFlashDraftModel = object()
    model_mod.extract_context_feature = object()
    model_mod.sample = sample
    monkeypatch.setitem(sys.modules, "dflash.model", model_mod)

    assert dflash.sample is sample


def test_lazy_model_exports_report_missing_optional_dependency():
    with pytest.raises(ModuleNotFoundError, match="torch"):
        _ = dflash.sample


def test_unknown_lazy_export_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        _ = dflash.missing
