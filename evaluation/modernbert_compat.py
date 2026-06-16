"""Shared ModernBert eval-loading shim.

transformers 5.11 ModernBert re-emits `rope_parameters` with extra top-level
keys on config reload, which huggingface_hub 1.19's strict-dataclass validator
rejects. The per-layer rope_theta still parses correctly, so we relax validation
for *that one field only* - every other field stays strictly validated, so a
genuinely malformed config still fails loudly.
"""


def patch_hf_validator() -> None:
    try:
        import huggingface_hub.dataclasses as hd
    except Exception:
        return
    orig = getattr(hd, "type_validator", None)
    if orig is None or getattr(orig, "_modernbert_patched", False):
        return

    def lenient(name, value, *args, **kwargs):
        try:
            return orig(name, value, *args, **kwargs)
        except Exception:
            if name == "rope_parameters":  # known ModernBert reload quirk
                return
            raise

    lenient._modernbert_patched = True
    hd.type_validator = lenient
