"""Check whether transformers_replace patches are installed."""


def check_whether_transformers_replace_is_installed_correctly() -> bool:
    """Return True if transformers has been patched with our custom modules."""
    try:
        # Check for patched Gemma with adaRMS support
        from transformers.models.gemma.configuration_gemma import GemmaConfig

        cfg = GemmaConfig()
        # Our patched config has use_adarms attribute
        if not hasattr(cfg, "use_adarms"):
            return False

        # Check for patched modeling_gemma with _gated_residual
        from transformers.models.gemma import modeling_gemma

        if not hasattr(modeling_gemma, "_gated_residual"):
            return False

        return True
    except Exception:
        return False
