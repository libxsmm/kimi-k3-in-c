"""vLLM general plugin: CPU versions of the Kimi K3 ops vLLM only ships for GPUs.

Loaded in every vLLM process through the ``vllm.general_plugins`` entry point.
On non-CPU platforms it does nothing.
"""

_applied = False


def register() -> None:
    global _applied
    if _applied:
        return
    from vllm.platforms import current_platform

    if not current_platform.is_cpu():
        return
    from . import patch

    patch.apply()
    import os

    if os.environ.get("K3CPU_PROF"):
        from . import prof

        prof.install()
    _applied = True
