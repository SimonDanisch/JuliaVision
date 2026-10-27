"""Drain a device and persist its pipeline cache before the worker is retired.

Cache persistence normally happens at clean process exit; retired workers are
killed, so save first rather than losing the warm pipeline cache.
"""
function finish_device!(device)
    Mantle.waitidle(device)
    Mantle.save_pipeline_cache!(device)
    nothing
end
