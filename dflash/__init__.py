__all__ = [
    "DFlash2DraftModel",
    "DFlashDraftModel",
    "extract_context_feature",
    "load_and_process_dataset",
    "sample",
]


def __getattr__(name):
    if name == "load_and_process_dataset":
        from .benchmark import load_and_process_dataset

        return load_and_process_dataset

    if name in {"DFlashDraftModel", "DFlash2DraftModel", "extract_context_feature", "sample"}:
        from .model import (
            DFlash2DraftModel,
            DFlashDraftModel,
            extract_context_feature,
            sample,
        )

        return {
            "DFlashDraftModel": DFlashDraftModel,
            "DFlash2DraftModel": DFlash2DraftModel,
            "extract_context_feature": extract_context_feature,
            "sample": sample,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
