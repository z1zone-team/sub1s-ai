"""Load exported components only when requested, keeping pure APIs independent of voice dependencies."""
from importlib import import_module

_EXPORTS = {'BriefingPipeline': ('briefing_pipeline', 'BriefingPipeline'), 'EmbeddingPipeline': ('embedding_pipeline', 'EmbeddingPipeline'), 'NluPipeline': ('nlu_pipeline', 'NluPipeline'), 'SttPipeline': ('stt_pipeline', 'SttPipeline'), 'stt_pipeline': ('stt_pipeline', 'stt_pipeline')}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
