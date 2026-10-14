"""Load exported components only when requested, keeping pure APIs independent of voice dependencies."""
from importlib import import_module

_EXPORTS = {'briefing_router': ('briefing_route', 'router'), 'health_router': ('health_route', 'router'), 'nlu_router': ('nlu_route', 'router'), 'preference_router': ('preference_route', 'router'), 'stt_router': ('stt_route', 'router'), 'voice_process_router': ('voice_process_route', 'router')}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
