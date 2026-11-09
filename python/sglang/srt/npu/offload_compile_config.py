"""Explicit GE compilation policy for alias-safe Offload graphs only."""
from torchair.configs._option_base import OptionValue
from torchair.configs.compiler_config import CompilerConfig


class AliasSafeCompilerConfig(CompilerConfig):
    """Serialize the real GE option, including it in TorchAir config identity."""

    def __init__(self):
        super().__init__()
        # This TorchAir version has no public ge_config cache-mode field.
        # Keep a typed leaf for its config equality/cache identity and translate
        # it to the supported GE global option in as_dict().
        object.__setattr__(self, "_offload_op_cache_mode", OptionValue("disable", ["disable"]))
        self._fixed_attrs.append("_offload_op_cache_mode")

    def as_dict(self):
        local_options, global_options = super().as_dict()
        local_options.pop("_offload_op_cache_mode", None)
        global_options["ge.op_compiler_cache_mode"] = self._offload_op_cache_mode.value
        return local_options, global_options


def require_compatible_ge_initialization():
    """GE cannot change global compile options after its first initialization."""
    from torchair.core import _backend

    previous = _backend._GLOBAL_COMPILE_OPTION
    if previous is not None and previous.get("ge.op_compiler_cache_mode") != "disable":
        raise RuntimeError(
            "Alias-safe Offload requires GE op compiler cache disabled before GE initialization; "
            "start a fresh worker with the Offload compiler configuration"
        )
