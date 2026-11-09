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
"""Launch Python host functions into a CUDA graph (capture + replay).

See ``_jit.py`` for the compilation model and usage notes. Public surface:

    from sglang.srt.hostfunc import hostfunc, launch_hostfunc
"""

from sglang.srt.hostfunc._jit import (
    free_all_hostfunc_user_data,
    free_hostfunc_user_data,
    hostfunc,
    launch_hostfunc,
)

__all__ = [
    "hostfunc",
    "launch_hostfunc",
    "free_hostfunc_user_data",
    "free_all_hostfunc_user_data",
]
