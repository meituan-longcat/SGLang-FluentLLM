# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""JIT loader + Python API for launching host functions into a CUDA graph.

Launch plain Python functions onto a CUDA stream so they can be *captured into*
and *replayed by* a CUDA graph. The C++ binding lives next to this file in
``_binding.cpp`` and is JIT-compiled here with ``torch.utils.cpp_extension.load``.

Compilation model (important):
  * This is JIT, but compiled-once-then-reused. The first import triggers a
    Ninja build into ``~/.cache/torch_extensions/.../hostfunc_ext.so``. On later
    imports torch hashes the source + build flags and, if unchanged, just loads
    the cached ``.so`` (no recompile). So there is NO separate build script.
  * Editing ``_binding.cpp`` changes the hash and triggers an automatic rebuild.
  * Fully-AOT artifacts shipped in a wheel (no compiler on the target machine)
    would instead need a setup.py with CUDAExtension -- out of scope here.

Usage:
    from sglang.srt.hostfunc import hostfunc

    @hostfunc
    def advance(state):
        state.step()          # pure CPU work, no CUDA API calls

    with torch.cuda.graph(g, stream=stream):
        advance(state)        # captured into the graph
    g.replay()                # advance() runs again on every replay
"""

from __future__ import annotations

import atexit
import pathlib
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = pathlib.Path(__file__).parent.resolve()
_CSRC = _THIS_DIR / "_binding.cpp"


@lru_cache(maxsize=1)
def _ext():
    return load(
        name="hostfunc_ext",
        sources=[str(_CSRC)],
        with_cuda=True,
        extra_ldflags=["-lcudart"],
        verbose=True,
    )


# Handles to user-data allocated during graph capture (freeUserData=False).
# These must be freed explicitly, otherwise they live until process exit.
_HANDLES: set[int] = set()


def launch_hostfunc(fn, *args, **kwargs):
    """Enqueue ``fn(*args, **kwargs)`` as a host callback on the current stream.

    If the current stream is capturing a graph, the callback is recorded into
    the graph and its user data is kept alive so it can be replayed; an integer
    handle is returned (also tracked in ``_HANDLES``). Otherwise the callback
    runs once, frees itself, and ``None`` is returned.

    The callback MUST NOT call any CUDA API (a cudaLaunchHostFunc constraint).
    """
    stream = torch.cuda.current_stream()
    is_capturing = torch.cuda.is_current_stream_capturing()
    handle = _ext().launch_hostfunc(stream.cuda_stream, not is_capturing, fn, *args, **kwargs)
    if is_capturing:
        assert handle is not None
        _HANDLES.add(handle)
    else:
        assert handle is None
    return handle


def hostfunc(fn):
    """Decorator turning a plain Python function into a CUDA host callback."""

    def wrapper(*args, **kwargs):
        return launch_hostfunc(fn, *args, **kwargs)

    return wrapper


def free_hostfunc_user_data(handle: int) -> None:
    """Free a single captured callback's user data and stop tracking it."""
    if handle not in _HANDLES:
        raise ValueError(f"Hostfunc user data handle {handle} not found.")
    _ext().free_hostfunc_user_data(handle)
    _HANDLES.remove(handle)


def free_all_hostfunc_user_data() -> None:
    """Free all captured callbacks' user data (called automatically at exit)."""
    for handle in _HANDLES:
        _ext().free_hostfunc_user_data(handle)
    _HANDLES.clear()


atexit.register(free_all_hostfunc_user_data)
